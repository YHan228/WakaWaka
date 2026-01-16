#!/usr/bin/env python3
"""
01_ingest_corpus.py - Ingest poems from various corpus sources.

This script fetches poems from configured sources and outputs them to
data/raw/poems.jsonl for further processing in the build pipeline.

Sources:
  - test: Small set of hardcoded classical poems for development (15 poems)
  - ogura100: Ogura Hyakunin Isshu (100 poems, embedded)
  - oncoj: ONCOJ corpus from GitHub (~4,900 Old Japanese poems)
  - lapis: Nichibunken Lapis waka database (scraper, supports collection filtering)
  - all: Combine ogura100 + oncoj + lapis for full curriculum
  - file: Load from a local JSON/JSONL file

Features:
  - Deduplication by text hash (both within source and across runs)
  - Rate-limited scraping with configurable sleep intervals
  - Collection filtering for Lapis (e.g., only 古今集, 新古今集, 万葉集)
  - Random sampling to avoid volume bias in anthologies

Usage:
  python scripts/01_ingest_corpus.py --source test --max-poems 10
  python scripts/01_ingest_corpus.py --source ogura100
  python scripts/01_ingest_corpus.py --source lapis --collections 古今集 --max-poems 300
  python scripts/01_ingest_corpus.py --source lapis --collections imperial --max-poems 1000
  python scripts/01_ingest_corpus.py --source all
"""

import argparse
import hashlib
import io
import json
import logging
import random
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterator
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# Project root for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from wakawaka.utils.treebank_parser import parse_oncoj_file, parse_simple_bracketed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Cache directory for downloaded files
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# Rate limiting settings (per RESOURCES.md)
DEFAULT_SLEEP = 1.0  # seconds between requests
USER_AGENT = "WakaDecoder/1.0 (Classical Japanese Poetry Learning; contact@example.com)"

# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


def make_poem_record(
    text: str,
    source: str,
    author: str | None = None,
    collection: str | None = None,
    source_id: str | None = None,
    source_url: str | None = None,
) -> dict:
    """
    Create a standardized poem record with a unique ID.

    Args:
        text: The poem text (Japanese characters).
        source: Source identifier (e.g., "ogura100", "lapis", "oncoj").
        author: Optional poet name.
        collection: Optional anthology/collection name.
        source_id: Optional ID from the original source.
        source_url: Optional URL to the source.

    Returns:
        A dict with poem_id, text, text_hash, source, and metadata fields.
    """
    # Generate stable ID from source + text hash
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    poem_id = f"{source}_{text_hash}"

    return {
        "poem_id": poem_id,
        "text": text,
        "text_hash": text_hash,
        "source": source,
        "source_id": source_id,
        "source_url": source_url,
        "author": author,
        "collection": collection,
        "ingested_at": datetime.now().isoformat(),
    }


# -----------------------------------------------------------------------------
# Source: test (hardcoded development poems)
# -----------------------------------------------------------------------------

TEST_POEMS = [
    {"text": "古池や蛙飛び込む水の音", "author": "松尾芭蕉", "collection": "haiku"},
    {"text": "閑さや岩にしみ入る蝉の声", "author": "松尾芭蕉", "collection": "haiku"},
    {"text": "夏草や兵どもが夢の跡", "author": "松尾芭蕉", "collection": "haiku"},
    {"text": "菜の花や月は東に日は西に", "author": "与謝蕪村", "collection": "haiku"},
    {"text": "春の海ひねもすのたりのたりかな", "author": "与謝蕪村", "collection": "haiku"},
    {"text": "やせ蛙負けるな一茶これにあり", "author": "小林一茶", "collection": "haiku"},
    {"text": "秋の田のかりほの庵の苫をあらみわが衣手は露にぬれつつ", "author": "天智天皇", "collection": "百人一首"},
    {"text": "春過ぎて夏来にけらし白妙の衣ほすてふ天の香具山", "author": "持統天皇", "collection": "百人一首"},
    {"text": "あしびきの山鳥の尾のしだり尾のながながし夜をひとりかも寝む", "author": "柿本人麻呂", "collection": "百人一首"},
    {"text": "田子の浦にうち出でてみれば白妙の富士の高嶺に雪は降りつつ", "author": "山部赤人", "collection": "百人一首"},
    {"text": "奥山に紅葉踏みわけ鳴く鹿の声きく時ぞ秋は悲しき", "author": "猿丸大夫", "collection": "百人一首"},
    {"text": "かささぎの渡せる橋におく霜の白きを見れば夜ぞ更けにける", "author": "中納言家持", "collection": "百人一首"},
    {"text": "花の色は移りにけりないたづらにわが身世にふるながめせしまに", "author": "小野小町", "collection": "古今和歌集"},
    {"text": "ひさかたの光のどけき春の日にしづ心なく花の散るらむ", "author": "紀友則", "collection": "古今和歌集"},
    {"text": "人はいさ心も知らずふるさとは花ぞ昔の香ににほひける", "author": "紀貫之", "collection": "古今和歌集"},
]


def load_test_poems(max_poems: int | None = None) -> Iterator[dict]:
    """
    Load hardcoded test poems for development.

    Args:
        max_poems: Maximum number of poems to return (default: all 15).

    Yields:
        Poem records with text, author, and collection.
    """
    poems = TEST_POEMS[:max_poems] if max_poems else TEST_POEMS
    for poem in poems:
        yield make_poem_record(
            text=poem["text"],
            source="test",
            author=poem.get("author"),
            collection=poem.get("collection"),
        )


# -----------------------------------------------------------------------------
# Source: ogura100 (Ogura Hyakunin Isshu - 100 poems)
# -----------------------------------------------------------------------------

OGURA_100 = [
    ("秋の田のかりほの庵の苫をあらみわが衣手は露にぬれつつ", "天智天皇"),
    ("春過ぎて夏来にけらし白妙の衣ほすてふ天の香具山", "持統天皇"),
    ("あしびきの山鳥の尾のしだり尾のながながし夜をひとりかも寝む", "柿本人麻呂"),
    ("田子の浦にうち出でてみれば白妙の富士の高嶺に雪は降りつつ", "山部赤人"),
    ("奥山に紅葉踏みわけ鳴く鹿の声きく時ぞ秋は悲しき", "猿丸大夫"),
    ("かささぎの渡せる橋におく霜の白きを見れば夜ぞ更けにける", "中納言家持"),
    ("天の原ふりさけ見れば春日なる三笠の山に出でし月かも", "安倍仲麿"),
    ("わが庵は都のたつみしかぞ住む世をうぢ山と人はいふなり", "喜撰法師"),
    ("花の色は移りにけりないたづらにわが身世にふるながめせしまに", "小野小町"),
    ("これやこの行くも帰るも別れては知るも知らぬも逢坂の関", "蝉丸"),
    ("わたの原八十島かけて漕ぎ出でぬと人には告げよ海人の釣舟", "参議篁"),
    ("天つ風雲の通ひ路吹きとぢよをとめの姿しばしとどめむ", "僧正遍昭"),
    ("筑波嶺の峰より落つるみなの川恋ぞつもりて淵となりぬる", "陽成院"),
    ("陸奥のしのぶもぢずり誰ゆゑに乱れそめにし我ならなくに", "河原左大臣"),
    ("君がため春の野に出でて若菜摘むわが衣手に雪は降りつつ", "光孝天皇"),
    ("立ち別れいなばの山の峰に生ふるまつとし聞かば今帰り来む", "中納言行平"),
    ("ちはやぶる神代も聞かず竜田川からくれなゐに水くくるとは", "在原業平朝臣"),
    ("住の江の岸による波よるさへや夢の通ひ路人目よくらむ", "藤原敏行朝臣"),
    ("難波潟短き蘆のふしの間も逢はでこの世を過ぐしてよとや", "伊勢"),
    ("わびぬれば今はた同じ難波なるみをつくしても逢はむとぞ思ふ", "元良親王"),
    ("今来むといひしばかりに長月の有明の月を待ち出でつるかな", "素性法師"),
    ("吹くからに秋の草木のしをるればむべ山風を嵐といふらむ", "文屋康秀"),
    ("月見ればちぢにものこそ悲しけれわが身一つの秋にはあらねど", "大江千里"),
    ("このたびは幣も取りあへず手向山紅葉の錦神のまにまに", "菅家"),
    ("名にし負はば逢坂山のさねかづら人に知られでくるよしもがな", "三条右大臣"),
    ("小倉山峰のもみぢ葉心あらば今ひとたびのみゆき待たなむ", "貞信公"),
    ("みかの原わきて流るるいづみ川いつ見きとてか恋しかるらむ", "中納言兼輔"),
    ("山里は冬ぞ寂しさまさりける人目も草もかれぬと思へば", "源宗于朝臣"),
    ("心あてに折らばや折らむ初霜のおきまどはせる白菊の花", "凡河内躬恒"),
    ("有明のつれなく見えし別れより暁ばかり憂きものはなし", "壬生忠岑"),
    ("朝ぼらけ有明の月と見るまでに吉野の里に降れる白雪", "坂上是則"),
    ("山川に風のかけたるしがらみは流れもあへぬ紅葉なりけり", "春道列樹"),
    ("ひさかたの光のどけき春の日にしづ心なく花の散るらむ", "紀友則"),
    ("誰をかも知る人にせむ高砂の松も昔の友ならなくに", "藤原興風"),
    ("人はいさ心も知らずふるさとは花ぞ昔の香ににほひける", "紀貫之"),
    ("夏の夜はまだ宵ながら明けぬるを雲のいづこに月宿るらむ", "清原深養父"),
    ("白露に風の吹きしく秋の野はつらぬきとめぬ玉ぞ散りける", "文屋朝康"),
    ("忘らるる身をば思はず誓ひてし人の命の惜しくもあるかな", "右近"),
    ("浅茅生の小野の篠原しのぶれどあまりてなどか人の恋しき", "参議等"),
    ("忍ぶれど色に出でにけりわが恋はものや思ふと人の問ふまで", "平兼盛"),
    ("恋すてふわが名はまだき立ちにけり人知れずこそ思ひそめしか", "壬生忠見"),
    ("契りきなかたみに袖をしぼりつつ末の松山波越さじとは", "清原元輔"),
    ("逢ひ見てののちの心にくらぶれば昔はものを思はざりけり", "権中納言敦忠"),
    ("逢ふことの絶えてしなくはなかなかに人をも身をも恨みざらまし", "中納言朝忠"),
    ("あはれともいふべき人は思ほえで身のいたづらになりぬべきかな", "謙徳公"),
    ("由良の門を渡る舟人かぢを絶え行くへも知らぬ恋の道かな", "曾禰好忠"),
    ("八重むぐら茂れる宿の寂しきに人こそ見えね秋は来にけり", "恵慶法師"),
    ("風をいたみ岩うつ波のおのれのみくだけて物を思ふころかな", "源重之"),
    ("御垣守衛士のたく火の夜は燃え昼は消えつつ物をこそ思へ", "大中臣能宣朝臣"),
    ("君がため惜しからざりし命さへ長くもがなと思ひけるかな", "藤原義孝"),
    ("かくとだにえやは伊吹のさしも草さしも知らじな燃ゆる思ひを", "藤原実方朝臣"),
    ("明けぬれば暮るるものとは知りながらなほ恨めしき朝ぼらけかな", "藤原道信朝臣"),
    ("嘆きつつひとり寝る夜の明くる間はいかに久しきものとかは知る", "右大将道綱母"),
    ("忘れじの行く末まではかたければ今日を限りの命ともがな", "儀同三司母"),
    ("滝の音は絶えて久しくなりぬれど名こそ流れてなほ聞こえけれ", "大納言公任"),
    ("あらざらむこの世のほかの思ひ出に今ひとたびの逢ふこともがな", "和泉式部"),
    ("めぐり逢ひて見しやそれともわかぬ間に雲がくれにし夜半の月かな", "紫式部"),
    ("有馬山猪名の笹原風吹けばいでそよ人を忘れやはする", "大弐三位"),
    ("やすらはで寝なましものをさ夜更けてかたぶくまでの月を見しかな", "赤染衛門"),
    ("大江山いく野の道の遠ければまだふみもみず天の橋立", "小式部内侍"),
    ("いにしへの奈良の都の八重桜けふ九重ににほひぬるかな", "伊勢大輔"),
    ("夜をこめて鳥のそら音ははかるともよに逢坂の関はゆるさじ", "清少納言"),
    ("今はただ思ひ絶えなむとばかりを人づてならでいふよしもがな", "左京大夫道雅"),
    ("朝ぼらけ宇治の川霧たえだえにあらはれわたる瀬々の網代木", "権中納言定頼"),
    ("恨みわび干さぬ袖だにあるものを恋に朽ちなむ名こそ惜しけれ", "相模"),
    ("もろともにあはれと思へ山桜花よりほかに知る人もなし", "前大僧正行尊"),
    ("春の夜の夢ばかりなる手枕にかひなく立たむ名こそ惜しけれ", "周防内侍"),
    ("心にもあらでうき世にながらへば恋しかるべき夜半の月かな", "三条院"),
    ("嵐吹く三室の山のもみぢ葉は竜田の川の錦なりけり", "能因法師"),
    ("寂しさに宿を立ち出でてながむればいづこも同じ秋の夕暮", "良暹法師"),
    ("夕されば門田の稲葉おとづれて蘆のまろやに秋風ぞ吹く", "大納言経信"),
    ("音に聞くたかしの浜のあだ波はかけじや袖のぬれもこそすれ", "祐子内親王家紀伊"),
    ("高砂の尾の上の桜咲きにけり外山の霞立たずもあらなむ", "前中納言匡房"),
    ("憂かりける人を初瀬の山おろしよはげしかれとは祈らぬものを", "源俊頼朝臣"),
    ("契りおきしさせもが露を命にてあはれ今年の秋もいぬめり", "藤原基俊"),
    ("わたの原漕ぎ出でて見ればひさかたの雲居にまがふ沖つ白波", "法性寺入道前関白太政大臣"),
    ("瀬をはやみ岩にせかるる滝川のわれても末に逢はむとぞ思ふ", "崇徳院"),
    ("淡路島通ふ千鳥の鳴く声に幾夜寝覚めぬ須磨の関守", "源兼昌"),
    ("秋風にたなびく雲の絶え間よりもれ出づる月の影のさやけさ", "左京大夫顕輔"),
    ("長からむ心も知らず黒髪の乱れて今朝はものをこそ思へ", "待賢門院堀河"),
    ("ほととぎす鳴きつる方をながむればただ有明の月ぞ残れる", "後徳大寺左大臣"),
    ("思ひわびさても命はあるものをうきに堪へぬは涙なりけり", "道因法師"),
    ("世の中よ道こそなけれ思ひ入る山の奥にも鹿ぞ鳴くなる", "皇太后宮大夫俊成"),
    ("ながらへばまたこのごろやしのばれむ憂しと見し世ぞ今は恋しき", "藤原清輔朝臣"),
    ("夜もすがらもの思ふころは明けやらで閨のひまさへつれなかりけり", "俊恵法師"),
    ("嘆けとて月やはものを思はするかこち顔なるわが涙かな", "西行法師"),
    ("村雨の露もまだ干ぬまきの葉に霧立ちのぼる秋の夕暮", "寂蓮法師"),
    ("難波江の蘆のかりねのひとよゆゑ身を尽くしてや恋ひわたるべき", "皇嘉門院別当"),
    ("玉の緒よ絶えなば絶えねながらへば忍ぶることの弱りもぞする", "式子内親王"),
    ("見せばやな雄島の海人の袖だにも濡れにぞ濡れし色はかはらず", "殷富門院大輔"),
    ("きりぎりす鳴くや霜夜のさむしろに衣かたしきひとりかも寝む", "後京極摂政前太政大臣"),
    ("わが袖は潮干に見えぬ沖の石の人こそ知らねかわく間もなし", "二条院讃岐"),
    ("世の中は常にもがもな渚漕ぐ海人の小舟の綱手かなしも", "鎌倉右大臣"),
    ("み吉野の山の秋風さ夜更けてふるさと寒く衣うつなり", "参議雅経"),
    ("おほけなく憂き世の民におほふかな我が立つ杣にすみ染の袖", "前大僧正慈円"),
    ("花さそふ嵐の庭の雪ならでふりゆくものはわが身なりけり", "入道前太政大臣"),
    ("来ぬ人を松帆の浦の夕なぎに焼くや藻塩の身もこがれつつ", "権中納言定家"),
    ("風そよぐ楢の小川の夕暮はみそぎぞ夏のしるしなりける", "従二位家隆"),
    ("人も惜し人も恨めしあぢきなく世を思ふゆゑにもの思ふ身は", "後鳥羽院"),
    ("ももしきや古き軒端のしのぶにもなほあまりある昔なりけり", "順徳院"),
]


def load_ogura100_poems(max_poems: int | None = None) -> Iterator[dict]:
    """
    Load Ogura Hyakunin Isshu (百人一首) poems.

    The 100 poems are embedded in this script with verified text and authors.
    These are canonical classical Japanese poems, ideal as a curriculum foundation.

    Args:
        max_poems: Maximum number of poems to return (default: all 100).

    Yields:
        Poem records numbered 1-100 with text, author, and collection="百人一首".
    """
    poems = OGURA_100[:max_poems] if max_poems else OGURA_100
    for i, (text, author) in enumerate(poems, 1):
        yield make_poem_record(
            text=text,
            source="ogura100",
            author=author,
            collection="百人一首",
            source_id=str(i),
        )


# -----------------------------------------------------------------------------
# Source: oncoj (ONCOJ corpus from GitHub)
# -----------------------------------------------------------------------------

ONCOJ_GITHUB_URL = "https://api.github.com/repos/ONCOJ/data/zipball/release"
ONCOJ_CACHE_FILE = "oncoj_release.zip"


def download_with_cache(url: str, cache_name: str, force: bool = False) -> bytes:
    """
    Download a file with local caching.

    Args:
        url: URL to download from.
        cache_name: Filename for the cached copy in data/cache/.
        force: If True, re-download even if cached.

    Returns:
        The file contents as bytes.

    Raises:
        URLError, HTTPError: If download fails.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / cache_name

    if cache_path.exists() and not force:
        logger.info(f"Using cached file: {cache_path}")
        return cache_path.read_bytes()

    logger.info(f"Downloading: {url}")
    req = Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urlopen(req, timeout=60) as response:
            data = response.read()
            cache_path.write_bytes(data)
            logger.info(f"Cached to: {cache_path}")
            return data
    except (URLError, HTTPError) as e:
        logger.error(f"Download failed: {e}")
        raise


def load_oncoj_poems(max_poems: int | None = None, force_download: bool = False) -> Iterator[dict]:
    """
    Load poems from ONCOJ (Oxford-NINJAL Corpus of Old Japanese).

    Downloads the corpus ZIP from GitHub (cached locally), then parses
    Penn Treebank format (.psd) files to extract poem texts.

    Args:
        max_poems: Maximum poems to return (default: all ~4,900).
        force_download: If True, re-download even if cached.

    Yields:
        Poem records with text, collection, source_id, and source_url.
    """
    try:
        zip_data = download_with_cache(ONCOJ_GITHUB_URL, ONCOJ_CACHE_FILE, force_download)
    except Exception as e:
        logger.error(f"Failed to download ONCOJ: {e}")
        logger.info("Try running with --force-download or check your network connection")
        return

    count = 0
    seen_texts = set()  # Deduplicate

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        # Find all .psd files in the archive (Penn treebank format)
        psd_files = [n for n in zf.namelist() if n.endswith('.psd')]
        logger.info(f"Found {len(psd_files)} PSD files in ONCOJ archive")

        for txt_file in psd_files:
            if max_poems and count >= max_poems:
                break

            try:
                content = zf.read(txt_file).decode('utf-8', errors='replace')
            except Exception as e:
                logger.warning(f"Failed to read {txt_file}: {e}")
                continue

            # Parse the file
            filename = Path(txt_file).name
            try:
                for parsed in parse_oncoj_file(content, filename):
                    if max_poems and count >= max_poems:
                        break

                    # Deduplicate by text
                    if parsed.text in seen_texts:
                        continue
                    seen_texts.add(parsed.text)

                    # Skip very short texts (likely fragments)
                    if len(parsed.text) < 10:
                        continue

                    yield make_poem_record(
                        text=parsed.text,
                        source="oncoj",
                        collection=parsed.metadata.get('collection_name', parsed.metadata.get('collection')),
                        source_id=parsed.text_id,
                        source_url=f"https://oncoj.ninjal.ac.jp/cgi-bin/oncoj.sh?search={parsed.text_id}",
                    )
                    count += 1

            except Exception as e:
                logger.warning(f"Failed to parse {txt_file}: {e}")
                # Try fallback parser
                try:
                    for parsed in parse_simple_bracketed(content, filename):
                        if max_poems and count >= max_poems:
                            break
                        if parsed.text in seen_texts:
                            continue
                        seen_texts.add(parsed.text)
                        if len(parsed.text) < 10:
                            continue
                        yield make_poem_record(
                            text=parsed.text,
                            source="oncoj",
                            source_id=parsed.text_id,
                        )
                        count += 1
                except Exception:
                    pass

    logger.info(f"Loaded {count} poems from ONCOJ")


# -----------------------------------------------------------------------------
# Source: lapis (Nichibunken Lapis waka database)
# -----------------------------------------------------------------------------

LAPIS_BASE_URL = "https://lapis.nichibun.ac.jp/waka"
LAPIS_INDEX_URLS = {
    "era": f"{LAPIS_BASE_URL}/index_era.html",
    "creator": f"{LAPIS_BASE_URL}/index_creator.html",
    "creation": f"{LAPIS_BASE_URL}/index_creation.html",
}


def fetch_with_rate_limit(url: str, sleep: float = DEFAULT_SLEEP) -> str:
    """
    Fetch URL content with rate limiting and proper User-Agent.

    Args:
        url: URL to fetch.
        sleep: Seconds to wait after each request (default: 1.0).

    Returns:
        The page content as a string.

    Raises:
        URLError, HTTPError: If fetch fails (after logging warning).
    """
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=30) as response:
            content = response.read().decode('utf-8', errors='replace')
        time.sleep(sleep)  # Rate limit
        return content
    except (URLError, HTTPError) as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        time.sleep(sleep * 2)  # Backoff on error
        raise


def extract_poems_from_lapis_work_page(html: str, url: str) -> Iterator[dict]:
    """
    Extract poems from a Lapis work page HTML.

    Handles two formats used by Lapis:
    - 古今集 style: kanji+hiragana text (e.g., "古池や蛙飛び込む水の音")
    - 新古今集/万葉集 style: hiragana with − separators (e.g., "あきの−たの−...")

    Args:
        html: Raw HTML content of the work page.
        url: URL of the page (stored in output for reference).

    Yields:
        Dicts with 'text', 'collection', and 'url' keys.
    """
    # Extract collection name from <h1>
    collection = None
    h1_match = re.search(r'<h1>(.*?)</h1>', html, re.DOTALL)
    if h1_match:
        collection = re.sub(r'<[^>]+>', '', h1_match.group(1)).strip()

    # Pattern 1: Kanji+hiragana text (古今集 style)
    pattern_kanji = r'<div[^>]*align="left"[^>]*>([ぁ-ん一-龯ァ-ヶー々]+)<br>'

    # Pattern 2: Hiragana with − separators (新古今集/万葉集 style)
    pattern_hira = r'<div[^>]*align="left"[^>]*>([ぁ-んァ-ヶー]+(?:−[ぁ-んァ-ヶー]+){2,})<br>'

    seen = set()

    # Try kanji pattern first
    for match in re.finditer(pattern_kanji, html, re.IGNORECASE):
        poem_text = match.group(1).strip().replace(' ', '').replace('\n', '')
        if 20 <= len(poem_text) <= 80 and poem_text not in seen:
            seen.add(poem_text)
            yield {"text": poem_text, "collection": collection, "url": url}

    # If no kanji matches, try hiragana pattern
    if not seen:
        for match in re.finditer(pattern_hira, html, re.IGNORECASE):
            poem_text = match.group(1).strip()
            # Remove − separators
            poem_text = poem_text.replace('−', '')
            if 25 <= len(poem_text) <= 50 and poem_text not in seen:
                seen.add(poem_text)
                yield {"text": poem_text, "collection": collection, "url": url}


# Collection names for the "imperial" shortcut (--collections imperial)
# Must match exact names as they appear in Lapis <h1> tags (and LAPIS_COLLECTION_PAGES keys)
LAPIS_IMPERIAL_COLLECTIONS = ["古今集", "新古今集", "万葉集・武田訓"]

# Direct page URLs for known major collections (fast access, no index scan needed)
# Keys are exact collection names as they appear in Lapis <h1> tags
LAPIS_COLLECTION_PAGES = {
    "古今集": "waka_i001.html",           # Kokin Wakashū (~1,100 poems)
    "新古今集": "waka_i010.html",         # Shin Kokin Wakashū (~2,000 poems)
    "万葉集・武田訓": "waka_i061.html",   # Man'yōshū - Takeda edition (~4,500 poems)
}


def load_lapis_poems(
    max_poems: int | None = None,
    sleep: float = DEFAULT_SLEEP,
    index_type: str = "era",
    collections: list[str] | None = None,
    random_sample: bool = True,
) -> Iterator[dict]:
    """
    Scrape poems from Nichibunken Lapis waka database.

    Poems are embedded directly in work pages (not separate poem pages).
    Respects rate limiting and robots.txt guidelines.

    When collections have known direct pages (古今集, 新古今集, 万葉集・武田訓),
    fetches only those pages for efficiency. Otherwise, scans the full index.

    Args:
        max_poems: Maximum poems to return (default: unlimited).
        sleep: Seconds to wait between requests (default: 1.0).
        index_type: Which index to scan - "era", "creator", or "creation".
        collections: Filter to exact collection names (e.g., ["古今集", "新古今集"]).
            Use exact names as they appear in Lapis <h1> tags.
        random_sample: If True, collect all matching poems first, then randomly
            sample to max_poems. Avoids bias toward early volumes. (default: True)

    Yields:
        Poem records with text, collection, and source_url.
    """
    if collections:
        logger.info(f"Starting Lapis scrape (collections: {collections}, max: {max_poems or 'unlimited'})")
    else:
        logger.info(f"Starting Lapis scrape (index: {index_type}, max: {max_poems or 'unlimited'})")

    count = 0
    seen_texts = set()
    errors = 0
    max_errors = 50  # Higher tolerance since we're scraping many pages

    try:
        # Fetch the index page
        index_url = LAPIS_INDEX_URLS.get(index_type, LAPIS_INDEX_URLS["era"])
        logger.info(f"Fetching index: {index_url}")
        index_html = fetch_with_rate_limit(index_url, sleep)

        # If all collections have known direct pages, skip index scan
        if collections and all(c in LAPIS_COLLECTION_PAGES for c in collections):
            work_links = [LAPIS_COLLECTION_PAGES[c] for c in collections]
            logger.info(f"Using direct pages for {collections}: {work_links}")
        else:
            # Extract links to work pages (waka_iXXX.html pattern)
            links = re.findall(r'href="(waka_i\d+\.html)"', index_html)
            work_links = list(set(links))  # Dedupe

            # If filtering by specific collections, prioritize known pages
            if collections:
                priority_pages = ["waka_i001.html", "waka_i010.html", "waka_i061.html"]
                priority = [p for p in priority_pages if p in work_links]
                others = [p for p in work_links if p not in priority_pages]
                random.shuffle(others)
                work_links = priority + others
            else:
                random.shuffle(work_links)

            logger.info(f"Found {len(work_links)} unique work pages")

        # Collect all matching poems first (for random sampling)
        all_poems = []

        for i, work_link in enumerate(work_links):
            # If not random sampling and we have enough, stop early
            if not random_sample and max_poems and count >= max_poems:
                break
            if errors >= max_errors:
                logger.error(f"Too many errors ({errors}), stopping")
                break

            work_url = f"{LAPIS_BASE_URL}/{work_link}"

            try:
                work_html = fetch_with_rate_limit(work_url, sleep)

                # Extract poems directly from the work page
                poems_on_page = 0
                for poem_data in extract_poems_from_lapis_work_page(work_html, work_url):
                    # Filter by collection if specified
                    collection_name = poem_data.get("collection", "")
                    if collections:
                        # Exact match: collection must equal one of the targets
                        if collection_name not in collections:
                            continue  # Skip poems not in target collections

                    if poem_data["text"] not in seen_texts:
                        seen_texts.add(poem_data["text"])
                        poem_record = make_poem_record(
                            text=poem_data["text"],
                            source="lapis",
                            collection=collection_name,
                            source_url=work_url,
                        )

                        if random_sample:
                            all_poems.append(poem_record)
                        else:
                            yield poem_record
                            count += 1
                            if max_poems and count >= max_poems:
                                break

                        poems_on_page += 1

                if poems_on_page > 0:
                    logger.debug(f"Extracted {poems_on_page} poems from {work_link} ({collection_name})")

                total_collected = len(all_poems) if random_sample else count
                if total_collected % 100 == 0 and total_collected > 0:
                    logger.info(f"Progress: {total_collected} poems scraped from {i+1}/{len(work_links)} pages")

            except Exception as e:
                errors += 1
                logger.warning(f"Error fetching {work_url}: {e}")

        # If random sampling, sample and yield
        if random_sample and all_poems:
            logger.info(f"Collected {len(all_poems)} poems, randomly sampling {min(max_poems or len(all_poems), len(all_poems))}")
            if max_poems and len(all_poems) > max_poems:
                sampled = random.sample(all_poems, max_poems)
            else:
                sampled = all_poems
                random.shuffle(sampled)  # Still shuffle for variety
            for poem in sampled:
                yield poem
                count += 1

    except Exception as e:
        logger.error(f"Failed to scrape Lapis: {e}")

    logger.info(f"Lapis scrape complete: {count} poems, {errors} errors")


# -----------------------------------------------------------------------------
# Source: file (load from local file)
# -----------------------------------------------------------------------------


def load_file_poems(input_path: Path, max_poems: int | None = None) -> Iterator[dict]:
    """
    Load poems from a local JSON or JSONL file.

    Args:
        input_path: Path to the input file (.json or .jsonl).
        max_poems: Maximum poems to return (default: all).

    Yields:
        Poem records. Input must have at least a 'text' field;
        other fields (author, collection, source_id, source_url) are optional.
    """
    count = 0
    with open(input_path, "r", encoding="utf-8") as f:
        if input_path.suffix == ".jsonl":
            for line in f:
                if max_poems and count >= max_poems:
                    break
                data = json.loads(line)
                yield make_poem_record(
                    text=data["text"],
                    source=data.get("source", "file"),
                    author=data.get("author"),
                    collection=data.get("collection"),
                    source_id=data.get("source_id"),
                    source_url=data.get("source_url"),
                )
                count += 1
        else:
            poems = json.load(f)
            for data in poems:
                if max_poems and count >= max_poems:
                    break
                yield make_poem_record(
                    text=data["text"],
                    source=data.get("source", "file"),
                    author=data.get("author"),
                    collection=data.get("collection"),
                    source_id=data.get("source_id"),
                    source_url=data.get("source_url"),
                )
                count += 1


# -----------------------------------------------------------------------------
# Source: all (combine multiple sources)
# -----------------------------------------------------------------------------


def load_all_poems(max_poems: int | None = None) -> Iterator[dict]:
    """
    Load poems from all available sources for full curriculum.

    Combines sources in priority order: ogura100 → oncoj → lapis.
    Deduplicates by text across all sources.

    Args:
        max_poems: Maximum total poems to return (default: unlimited).

    Yields:
        Poem records from all sources, deduplicated.
    """
    count = 0
    seen_texts = set()

    # Helper to dedupe and count
    def emit(poem: dict) -> dict | None:
        nonlocal count
        if max_poems and count >= max_poems:
            return None
        if poem["text"] in seen_texts:
            return None
        seen_texts.add(poem["text"])
        count += 1
        return poem

    # 1. Ogura 100 (high quality, known poems)
    logger.info("Loading Ogura 100...")
    for poem in load_ogura100_poems():
        result = emit(poem)
        if result:
            yield result
        if max_poems and count >= max_poems:
            return

    # 2. ONCOJ (main corpus, ~2000+ poems)
    logger.info("Loading ONCOJ...")
    for poem in load_oncoj_poems():
        result = emit(poem)
        if result:
            yield result
        if max_poems and count >= max_poems:
            return

    # 3. Lapis (supplemental, scrape 500 more if needed)
    if not max_poems or count < max_poems:
        remaining = (max_poems - count) if max_poems else 500
        logger.info(f"Loading Lapis (up to {remaining} more)...")
        for poem in load_lapis_poems(max_poems=remaining):
            result = emit(poem)
            if result:
                yield result
            if max_poems and count >= max_poems:
                return

    logger.info(f"Total poems loaded: {count}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Ingest poems from various corpus sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Development testing
  python scripts/01_ingest_corpus.py --source test --max-poems 5

  # Load Ogura Hyakunin Isshu (100 canonical poems)
  python scripts/01_ingest_corpus.py --source ogura100

  # Scrape specific collections from Lapis
  python scripts/01_ingest_corpus.py --source lapis --collections 古今集 --max-poems 300
  python scripts/01_ingest_corpus.py --source lapis --collections imperial --max-poems 1000

  # Full corpus from all sources
  python scripts/01_ingest_corpus.py --source all
        """,
    )
    parser.add_argument(
        "--source",
        choices=["test", "ogura100", "oncoj", "lapis", "all", "file"],
        default="test",
        help="Corpus source to ingest from (default: test)",
    )
    parser.add_argument(
        "--max-poems",
        type=int,
        default=None,
        help="Maximum number of poems to ingest (default: all)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input file path (required for --source file)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "poems.jsonl",
        help="Output JSONL file path (default: data/raw/poems.jsonl)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP,
        help=f"Sleep between requests for scraping (default: {DEFAULT_SLEEP}s)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of cached files",
    )
    parser.add_argument(
        "--collections",
        type=str,
        default=None,
        help="Filter lapis by collection prefixes, comma-separated (e.g., '古今,新古今,万葉'). Use 'imperial' for the three major anthologies.",
    )

    args = parser.parse_args()

    # Validate args
    if args.source == "file" and not args.input:
        parser.error("--input is required when --source is 'file'")

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load poems from source
    if args.source == "test":
        poems = load_test_poems(args.max_poems)
    elif args.source == "ogura100":
        poems = load_ogura100_poems(args.max_poems)
    elif args.source == "oncoj":
        poems = load_oncoj_poems(args.max_poems, args.force_download)
    elif args.source == "lapis":
        # Parse collections filter
        collections = None
        if args.collections:
            if args.collections.lower() == "imperial":
                collections = LAPIS_IMPERIAL_COLLECTIONS  # 古今集, 新古今集, 万葉集
            else:
                collections = [c.strip() for c in args.collections.split(",")]
        poems = load_lapis_poems(args.max_poems, args.sleep, collections=collections)
    elif args.source == "all":
        poems = load_all_poems(args.max_poems)
    elif args.source == "file":
        poems = load_file_poems(args.input, args.max_poems)
    else:
        parser.error(f"Unknown source: {args.source}")

    # Write to output (with deduplication by text_hash)
    count = 0
    skipped = 0
    seen_hashes = set()

    # If appending to existing file, load existing hashes
    if args.output.exists():
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing = json.loads(line)
                    seen_hashes.add(existing.get("text_hash", ""))
                except json.JSONDecodeError:
                    pass
        logger.info(f"Loaded {len(seen_hashes)} existing poems for deduplication")

    # Write new poems (append mode if file exists and has content)
    mode = "a" if seen_hashes else "w"
    with open(args.output, mode, encoding="utf-8") as f:
        for poem in poems:
            if poem["text_hash"] in seen_hashes:
                skipped += 1
                continue
            seen_hashes.add(poem["text_hash"])
            f.write(json.dumps(poem, ensure_ascii=False) + "\n")
            count += 1

    print(f"Ingested {count} poems from '{args.source}' to {args.output}")
    if skipped:
        print(f"  (skipped {skipped} duplicates)")

    # Warn if corpus is too small
    if count < 1500 and args.source == "all":
        print(f"WARNING: Only {count} poems. Target is 1500+ for good curriculum coverage.")


if __name__ == "__main__":
    main()

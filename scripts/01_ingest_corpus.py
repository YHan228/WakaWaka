#!/usr/bin/env python3
"""
01_ingest_corpus.py - Ingest poems from various corpus sources.

This script fetches poems from configured sources and outputs them to
data/raw/poems.jsonl for further processing in the build pipeline.

Sources:
  - test: Small set of hardcoded classical poems for development
  - ogura100: Ogura Hyakunin Isshu (100 poems) from Japanese text file
  - file: Load from a local JSON/JSONL file

Usage:
  python scripts/01_ingest_corpus.py --source test --max-poems 10
  python scripts/01_ingest_corpus.py --source ogura100
  python scripts/01_ingest_corpus.py --source file --input path/to/poems.jsonl
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator

# Project root for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

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
    """Create a standardized poem record."""
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

# Famous classical Japanese poems for development/testing
TEST_POEMS = [
    # Matsuo Basho - haiku
    {
        "text": "古池や蛙飛び込む水の音",
        "author": "松尾芭蕉",
        "collection": "haiku",
    },
    {
        "text": "閑さや岩にしみ入る蝉の声",
        "author": "松尾芭蕉",
        "collection": "haiku",
    },
    {
        "text": "夏草や兵どもが夢の跡",
        "author": "松尾芭蕉",
        "collection": "haiku",
    },
    # Yosa Buson - haiku
    {
        "text": "菜の花や月は東に日は西に",
        "author": "与謝蕪村",
        "collection": "haiku",
    },
    {
        "text": "春の海ひねもすのたりのたりかな",
        "author": "与謝蕪村",
        "collection": "haiku",
    },
    # Kobayashi Issa - haiku
    {
        "text": "やせ蛙負けるな一茶これにあり",
        "author": "小林一茶",
        "collection": "haiku",
    },
    # Hyakunin Isshu samples
    {
        "text": "秋の田のかりほの庵の苫をあらみわが衣手は露にぬれつつ",
        "author": "天智天皇",
        "collection": "百人一首",
    },
    {
        "text": "春過ぎて夏来にけらし白妙の衣ほすてふ天の香具山",
        "author": "持統天皇",
        "collection": "百人一首",
    },
    {
        "text": "あしびきの山鳥の尾のしだり尾のながながし夜をひとりかも寝む",
        "author": "柿本人麻呂",
        "collection": "百人一首",
    },
    {
        "text": "田子の浦にうち出でてみれば白妙の富士の高嶺に雪は降りつつ",
        "author": "山部赤人",
        "collection": "百人一首",
    },
    {
        "text": "奥山に紅葉踏みわけ鳴く鹿の声きく時ぞ秋は悲しき",
        "author": "猿丸大夫",
        "collection": "百人一首",
    },
    {
        "text": "かささぎの渡せる橋におく霜の白きを見れば夜ぞ更けにける",
        "author": "中納言家持",
        "collection": "百人一首",
    },
    # Kokinshū samples
    {
        "text": "花の色は移りにけりないたづらにわが身世にふるながめせしまに",
        "author": "小野小町",
        "collection": "古今和歌集",
    },
    {
        "text": "ひさかたの光のどけき春の日にしづ心なく花の散るらむ",
        "author": "紀友則",
        "collection": "古今和歌集",
    },
    {
        "text": "人はいさ心も知らずふるさとは花ぞ昔の香ににほひける",
        "author": "紀貫之",
        "collection": "古今和歌集",
    },
]


def load_test_poems(max_poems: int | None = None) -> Iterator[dict]:
    """Load hardcoded test poems."""
    poems = TEST_POEMS[:max_poems] if max_poems else TEST_POEMS
    for poem in poems:
        yield make_poem_record(
            text=poem["text"],
            source="test",
            author=poem.get("author"),
            collection=poem.get("collection"),
        )


# -----------------------------------------------------------------------------
# Source: ogura100 (Ogura Hyakunin Isshu)
# -----------------------------------------------------------------------------

# Full Ogura Hyakunin Isshu (100 poems)
# Source: Public domain classical anthology
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
    """Load Ogura Hyakunin Isshu poems."""
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
# Source: file (load from local file)
# -----------------------------------------------------------------------------


def load_file_poems(
    input_path: Path, max_poems: int | None = None
) -> Iterator[dict]:
    """Load poems from a local JSON or JSONL file."""
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
            # Assume JSON array
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
# Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Ingest poems from various corpus sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/01_ingest_corpus.py --source test --max-poems 5
  python scripts/01_ingest_corpus.py --source ogura100
  python scripts/01_ingest_corpus.py --source file --input poems.jsonl
        """,
    )
    parser.add_argument(
        "--source",
        choices=["test", "ogura100", "file"],
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
    elif args.source == "file":
        poems = load_file_poems(args.input, args.max_poems)
    else:
        parser.error(f"Unknown source: {args.source}")

    # Write to output
    count = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for poem in poems:
            f.write(json.dumps(poem, ensure_ascii=False) + "\n")
            count += 1

    print(f"Ingested {count} poems from '{args.source}' to {args.output}")


if __name__ == "__main__":
    main()

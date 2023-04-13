#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:46:30 2023

@author: hyeontaemin
"""

tagling = ["아보카도",
"오렌지",
"칵테일토마토",
"복숭아",
"참외",
"스위티오",
"블루베리",
"토마토",
"딸기",
"냉동딸기",
"감귤",
"수라향",
"천혜향",
"한라봉",
"스테비아",
"방울토마토",
"메론",
"대추토마토",
"골든키위",
"배",
"사과",
"냉동블루베리",
"용과",
"레몬",
"못난이사과",
"애플망고",
"망고",
"포도",
"거봉",
"청포도",
"샤인머스켓",
"탐라향",
"그린키위",
"황금향",
"귤",
"신비향",
"카라향",
"곶감",
"감",
"패션후르츠",
"라즈베리",
"홍시",
"리치",
"두리안",
"람푸탄",
"무지개망고",
"수박",
"백향과",
"트리플베리",
"망고스틴",
"바나나",
"자몽",
"황도",
"백도",
"라임",
"코코넛",
"파파야",
"복분자",
"밤",
"자두",
"람부탄",
"체리",
"호박",
"도라지",
"더덕",
"고구마",
"두릅",
"당근",
"콜라비",
"꿀고구마",
"호박고구마",
"밤고구마",
"쑥",
"감자",
"배추",
"옥수수",
"찰옥수수",
"바질",
"여주",
"단호박",
"표고버섯",
"파프리카",
"양파",
"홍감자",
"생마",
"참마",
"미나리",
"비트",
"케일",
"샐러리",
"마늘",
"풋마늘",
"고추",
"깻잎",
"상추",
"명이나물",
"냉이",
"쪽파",
"느타리버섯",
"곤드레",
"양배추",
"오이",
"연근",
"양송이버섯",
"알로에",
"목이버섯",
"흰목이버섯",
"시금치",
"브로콜리",
"야콘",
"파",
"대파",
"고사리",
"우거지",
"생강",
"부추",
"땡초",
"갓",
"얼갈이배추",
"시래기",
"청양고추",
"송이버섯",
"새송이버섯",
"오이고추",
"건표고버섯",
"풋고추",
"꽈리고추",
"고수",
"무",
"알타리무",
"아스파라거스",
"석이버섯",
"적겨자",
"적근대",
"치커리",
"우엉",
"능이버섯",
"무순",
"봄동",
"녹차",
"상황버섯",
"백오이",
"열무",
"루꼴라",
"돌산갓",
"다진마늘",
"자색고구마",
"피망",
"할라피뇨",
"올리브",
"양상추",
"마늘쫑",
"흑마늘",
"두부",
"포두부",
"애호박",
"취나물",
"홍고추",
"검은콩",
"땅콩",
"완두콩",
"콩",
"백다다기",
"토란",
"다다기오이",
"고구마줄기",
"연두부",
"당조고추",
"갯나물",
"당귀잎",
"돼지감자",
"쑥갓",
"참나물",
"애플민트",
"민트",
"칡",
"고구마순",
"숙주나물",
"콩나물",
"청경채",
"깐마늘",
"초당두부",
"오크라",
"송화버섯",
"크로와상",
"고구마빵",
"현미빵",
"포켓몬빵",
"통밀식빵",
"통밀빵",
"감자빵",
"브라운브레드",
"크로플",
"도넛",
"풀빵",
"스콘",
"생지",
"크림빵",
"뉴욕치즈케이크",
"치아바타",
"보리빵",
"옥수수빵",
"막걸리빵",
"팥빵",
"플레인치아바타",
"복숭아빵",
"찐빵",
"베이글",
"부산바다샌드",
"안흥찐빵",
"호밀빵",
"바게트",
"깜빠뉴",
"찰보리빵",
"경주빵",
"우리쌀 카스테라",
"뉴욕샌드위치식빵",
"샌드위치",
"허니브레드",
"토스트",
"카스테라",
"대왕카스테라",
"밤만쥬",
"마늘바게트",
"모닝빵",
"단판빵",
"마늘빵",
"마들렌",
"오븐에구운도넛",
"붕어빵",
"조각케이크",
"파이",
"꿀호떡",
"와플",
"커피번",
"로티번",
"치즈번",
"모카번",
"호두과자",
"호두파이",
"삼림호빵",
"아이러브 토스트",
"밤식빵",
"초코파이",
"수제파이",
"뚱카롱",
"마카롱",
"크로크무슈",
"대만 샌드위치",
"맘모스빵",
"버터롤",
"커피콩빵",
"페스츄리",
"대추빵",
"보름달",
"델리만쥬",
"통밀 베이글",
"햄버거빵",
"초코롤",
"에그타르트",
"우우빵",
"허니버터브레드",
"프레즐",
"커클랜드 베이글",
"크림치즈프레즐",
"필라델피아 치즈케익",
"필라델피아 치즈케익 플레인",
"치즈케이크",
"건빵",
"버터크라상",
"경주보리빵",
"롤케이크",
"비건빵",
"도너츠",
"순수생크림 카스테라",
"술빵",
"맘모스주니어",
"도라야끼",
"초코머핀",
"머핀",
"알밤단판빵",
"핫도그",
"피자빵",
"치즈식빵",
"두부도너츠",
"달광도넛",
"소금빵",
"밤파이",
"땅콩샌드",
"소보로",
"땅콩센드",
"부시맨빵",
"슈크림빵",
"우유식빵",
"식빵",
"커스타드",
"쌀식빵",
"티라미수",
"어니언치즈빵",
"파운드케이크",
"밀크요팡",
"오스트콘",
"허니케이크",
"레몬파운드케이크",
"브리오쉬번",
"버거",
"햄버거",
"국화빵",
"십원빵",
"수제버거",
"쿠키",
"오예스",
"몽쉘",
"정통보름달",
"애플시나몬롤빵",
"깨방정 스틱",
"바나나빵",
"고구마롤빵",
"비빔밥빵",
"딸기케이크",
"당근케이크",
"초코케이크",
"빠네빵",
"땅콩빵",
"우유생크림빵",
"생크림도넛",
"초코와플",
"바스트케이크",
"유자",
"도제식빵",
"우유롤케익",
"완두앙금",
"약과",
"오색송편",
"수수팥떡",
"백설기",
"꿀설기",
"떡쌈",
"모찌떡",
"찰떡",
"인절미",
"영양찰떡",
"영양떡",
"호박떡",
"쑥인절미",
"오메기떡",
"가래떡",
"약밥",
"백일떡",
"보리떡",
"소떡소떡",
"현미보리떡",
"치즈떡",
"크림떡",
"통밤우유설기",
"깨송편",
"쑥떡",
"콩떡",
"우유떡",
"떡볶이떡",
"찰보리떡",
"모시떡",
"술떡",
"우유백설기",
"당고",
"떡국떡",
"떡국",
"우유설기",
"콩고물",
"꿀떡",
"조랭이떡",
"호떡",
"현미떡",
"호두찹살떡",
"쑥설기",
"모시송편",
"빙수떡",
"우유꿀백설기",
"구름떡",
"수능떡",
"시루떡",
"콩쑥개떡",
"치즈설기",
"경단",
"오색경단",
"치즈앙금떡",
"바나나떡",
"감자떡",
"고구마떡",
"절편",
"딸기망개떡",
"망개떡",
"씨앗호떡",
"연유",
"구워먹는치즈",
"스트링 치즈",
"무염버터",
"모짜렐라 치즈",
"슈레드",
"생크림",
"기버터",
"마아가린",
"골든치즈",
"코나도치즈",
"피자치즈",
"크림치즈",
"휘핑크림",
"고메버터",
"체다치즈",
"꽃소금",
"카이막",
"버터후레시",
"슬라이스치즈",
"스트링치즈",
"마가린",
"치즈케익",
"가염버터",
"앵커버터",
"파마산치즈",
"플레인치즈",
"쿠킹크림",
"브라운치즈",
"동물성 생크림",
"식물성 생크림",
"마스카포네치즈가루",
"그릴치즈",
"과일치즈",
"펠렛치즈",
"드라이버트",
"딸기잼",
"마스카포네치즈",
"사워크림",
"모찌리도후",
"바지페스토",
"식물성 휘핑크림",
"동물성 휘핑크림",
"호두",
"파운드마가린",
"체리페퍼크림치즈",
"다시마버터",
"고르곤졸라 치즈",
"롤치즈",
"콘맥앤치즈",
"체리페퍼",
"찢어먹는치즈",
"샐러드치즈",
"서울우유",
"가당연유",
"옥수수가루",
"마약옥수수가루",
"코코아 버터",
"나쵸치즈",
"다크초콜릿",
"아메리칸치즈",
"고단백 치즈",
"치즈분말",
"포션치즈",
"갈릭크림치즈",
"롤버터",
"바질크림치즈",
"칼슘치즈",
"맥앤치즈",
"보코치니",
"포션버터",
"라클렛치즈",
"유정란",
"청란",
"청계란",
"계란",
"초란",
"대란",
"왕란",
"특란",
"반숙란",
"훈제란",
"달걀",
"구운달걀",
"삶은달걀",
"오리알",
"타조알",
"거위알",
"민어",
"고등어",
"순살고등어",
"말린 갈치",
"갈치",
"전갱이",
"빙어",
"새우",
"삼치",
"가오리",
"가오리채",
"건대구",
"생대구",
"적돔",
"메로",
"간고등어",
"은갈치",
"메기",
"메로구이",
"도미",
"역돔",
"동태",
"가자미",
"아귀",
"임연수",
"볼락",
"조기",
"틸라피아",
"연어",
"우럭",
"자반고등어",
"참돔",
"양미리",
"참가자미",
"열빙어",
"꽁치",
"간재미",
"간제미",
"미꾸라지",
"숭어",
"대구고니",
"대구알",
"대구",
"간자미",
"참조기",
"장어",
"농어",
"아귀간",
"멸치",
"도루묵",
"양태",
"광어",
"도다리",
"빙어",
"달고기",
"고래고기",
"산낙지",
"문어",
"오징어",
"낙지",
"참치알",
"참치",
"황가오리",
"박대",
"잉어",
"열기",
"왕갈치",
"코다리",
"도다리",
"전어",
"병어",
"황태채",
"황태포",
"건오징어",
"아귀포",
"먹태채",
"복어",
"북어",
"먹태",
"오징어입",
"쥐포",
"먹태구이",
"북어포",
"가문어",
"가문어",
"명태",
"황태껍질",
"학꽁치포",
"쭈꾸미",
"황태",
"말린홍합",
"건홍합",
"홍합",
"북어껍질",
"참고구마",
"명태포",
"물메기채",
"준치오징어",
"노가리",
"진미채",
"장족",
"명엽체",
"명태알포",
"배오징어",
"학꽁치",
"세멸치",
"고바멸치",
"가리비",
"한치",
"꼬깔콘",
"꼬북칩",
"콘칩",
"브이콘",
"칙촉",
"빼빼로",
"포카칩",
"자갈치",
"새우깡",
"포스틱",
"꿀꽈배기",
"홈런볼",
"카스타드",
"쌀과자",
"보리과자",
"에낙",
"짱구",
"초코비",
"허니버터칩",
"초코송이",
"참쌀선과",
"참붕어빵",
"나쵸",
"빅파이",
"감자칩",
"예감",
"참깨스틱",
"바나나킥",
"눈을감자",
"소금",
"꽃게액젓",
"꽃게액",
"검정약콩",
"참치액",
"고춧가루",
"간장",
"청국장",
"가다랑어",
"사과식초",
"된장",
"맛간장",
"진간장",
"어간장",
"홍게간장",
"국간장",
"낫또",
"백설탕",
"흑설탕",
"고추장",
"설탕",
"죽염",
"생와사비",
"다시다",
"와사비",
"양조간장",
"쌈장",
"치킨파우더",
"쇠고기 다시다",
"라면스프",
"멸치육수",
"집된장",
"사골분말",
"후리가께",
"메주",
"가쓰오부시",
"제주콩",
"굵은소금",
"후추",
"양조식초",
"순후추",
"양념쌈장",
"고추냉이",
"알룰로스",
"자일로스설탕",
"고추가루",
"허브솔트",
"맛다시",
"밀가루",
"쌀",
"깨소금",
"미원",
"메주가루",
"환만식초",
"겨자",
"머스타드",
"해선간장",
"조청",
"물엿",
"맛술",
"춘장",
"볶음춘장",
"쫄면소스",
"조청쌀엿",
"도라지조청",
"각설탕",
"치킨스톡",
"돼지갈비소스",
"돼지갈비양념",
"맛소금",
"냉면장",
"된장찌개양념",
"멸치다시마",
"꼬막장",
"올리고당",
"초장",
"조개다시다",
"발사믹식초",
"청양고추가루",
"화이트 식초",
"강된장",
"연두",
"연겨자",
"양파가루",
"양파분말",
"볶음고추장",
"튀긴마늘",
"사탕수수",
"초피액젓",
"초고추장",
"화이트 발사믹",
"소갈비소스",
"소갈비양념",
"파슬리",
"스노윙시즈닝",
"시나몬",
"계피가루",
"콩된장",
"막장",
"돼지불고기소스",
"마파두부소스",
"마파두부양념",
"돼지불고기양념",
"쯔유",
"혼다시",
"순두부찌개양념",
"미향",
"부대찌개양념",
"월계수잎",
"쇠고기볶음양념",
"통후추",
"회초장",
"보리된장",
"비빔장",
"사카린",
"멸치액젓",
"오트밀",
"올리브오일",
"식용유",
"서리태",
"귀리",
"참기름",
"콩식용유",
"화유",
"올리브유",
"콩가루",
"고추기름",
"포도씨유",
"검정콩가루",
"아보카도오일",
"들기름",
"들꺠가루",
"코코넛오일",
"튀김가루",
"맛기름",
"옥수수전분",
"메밀가루",
"볶음콩가루",
"도토리묵가루",
"부침가루",
"아몬드가루",
"통참깨",
"파기름",
"빵가루",
"대파기름",
"베이킹파우더",
"아몬드파우더",
"카놀라유",
"올리브오일",
"박력분",
"중력분",
"강력분",
"트러플",
"쌀가루",
"카스테라가루",
"참깨",
"쌀",
"맵쌀",
"트러플오일",
"흑임자가루",
"감자전분",
"땅콩가루",
"현미유",
"콩식용유",
"검은깨가루",
"통밀가루",
"청국장가루",
"카나비노이드 오일",
"전분가루",
"꿀",
"벌꿀",
"식혜",
"해바라기유",
"쌀누룩",
"찹쌀가루",
"고구마가루",
"자색고구마가루",
"아몬드가루",
"코코넛분말",
"식빵가루",
"누룩",
"옥수수유",
"강황가루",
"메밀묵가루",
"짜장가루",
"카레가루",
"참쑥가루",
"송로버섯오일",
"송로버섯가루",
"송로버섯",
"찰밀가루",
"흑미",
"백미",
"햅쌀",
"현미",
"수향미",
"찹쌀",
"들깨",
"강낭콩",
"메주콩",
"팥",
"치아씨다",
"볶음참깨",
"메밀쌀",
"찰보리쌀",
"율무",
"잣",
"생땅콩",
"피스타치오",
"해바라기씨",
"캐슈넛",
"대추",
"대추칩",
"건자두",
"단밤",
"약단밤",
"맛밤",
"건망고",
"마카다미아",
"대추야자",
"호박씨",
"황잣",
"백잣",
"무화과",
"자몽칩",
"라임칩",
"레몬칩",
"헤이즐넛",
"바나나칩",
"건크랜베리",
"크랜베리",
"은행",
"건포도",
"감말랭이",
"통대창",
"편육",
"소대창",
"폭립",
"대창",
"막창",
"염통",
"소곱창",
"돼지막창",
"닭갈비",
"닭발",
"뽕잎",
"함박스테이크",
"떡갈비",
"오돌뼈",
"닭가슴살",
"한우곱창",
"족발",
"보쌈",
"팝콘치킨",
"돼지곱창",
"돼지막창",
"양대창",
"앞다리살",
"수육",
"치킨텐더",
"우삼겹",
"오겹살",
"삼겹살",
"차돌박이",
"욱회",
"봉",
"윙",
"닭날개",
"닭다리",
"닭다리살",
"곱창",
"로스트비프",
"부채살",
"오리고기",
"버팔로치킨",
"칠면조",
"살치살",
"양고기",
"양지머리",
"오징어무침회",
"고다리조림",
"충무김밥",
"감자조림",
"깻잎장아찌",
"쌀게무침",
"통도라지무침",
"고추무침",
"파래무침",
"명이장아찌",
"갈치조림",
"닭가슴살장조림",
"양념게장",
"초생강",
"치킨무",
"표고버섯볶음",
"황태껍질튀각",
"낙지젓갈",
"더덕구이",
"매실절임",
"시금치무침",
"콩자반",
"꽈리멸치볶음",
"흑임자연근무침",
"매실피클",
"락교",
"고들빼기무침",
"더덕양념구이",
"명태회무침",
"마늘쫑무침",
"다시마부각",
"콩조림",
"가지볶음",
"홍어무침",
"홍허회무침",
"오징어조림",
"멸치조림",
"멸치볶음",
"땅콩조림",
"땅콩볶음",
"연근조림",
"더덕무침",
"고추장아찌",
"마늘장아찌",
"메추리알장조림",
"고추부각",
"쌈무",
"마늘쫑무침",
"오이지무침",
"삭힌고추무침",
"유자단무지",
"매실장아찌",
"진미채무침",
"고추장멸치볶음",
"꼴뚜기조림",
"꼬들단무지",
"튀각",
"도토리묵",
"초무침",
"단무지",
"소고기장조림",
"돼지고기장조림",
"오이피클",
"김부각",
"장아찌",
"콩잎지",
"무말랭이"
]


sortedTag = sorted(tagling)

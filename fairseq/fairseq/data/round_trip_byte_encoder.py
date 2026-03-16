PSEUDO_BYTES = "一丁丂七丄丅丆万丈三上下丌不与丏丐丑丒专且丕世丗丘丙业丛东丝丞丟丠両丢丣两严並丧丨丩个丫丬中丮丯丰丱串丳临丵丶丷丸丹为主丼丽举丿乀乁乂乃乄久乆乇么义乊之乌乍乎乏乐乑乒乓乔乕乖乗乘乙乚乛乜九乞也习乡乢乣乤乥书乧乨乩乪乫乬乭乮乯买乱乲乳乴乵乶乷乸乹乺乻乼乽乾乿亀亁亂亃亄亅了亇予争亊事二亍于亏亐云互亓五井亖亗亘亙亚些亜亝亞亟亠亡亢亣交亥亦产亨亩亪享京亭亮亯亰亱亲亳亴亵亶亷亸亹人亻亼亽亾亿什仁仂仃仄仅仆仇仈仉今介仌仍从仏仐仑仒仓仔仕他仗付仙仚仛仜仝仞仟仠仡仢代令以仦仧仨仩仪仫们仭仮仯仰仱仲仳仴仵件价仸仹仺任仼份仾仿"
PSEUDO_BYTES_TO_BYTE = {cp: i for (i, cp) in enumerate(PSEUDO_BYTES)}

def pseudo_bytes_to_byte_list(pseudo_bytes):
    return [PSEUDO_BYTES_TO_BYTE.get(b, 0) for b in pseudo_bytes]

def encode_as_pseudo_bytes(s):
    return ''.join(PSEUDO_BYTES[b] for b in s.encode())

def decode_pseudo_bytes_to_string(pseudo_bytes):
    return bytes(pseudo_bytes_to_byte_list(pseudo_bytes)).decode('utf-8')


def _decode_pseudo_bytes_to_string(pseudo_bytes):
    try:
        return bytes(pseudo_bytes_to_byte_list(pseudo_bytes)).decode('utf-8')
    except:
        return pseudo_bytes

def encode_as_pseudo_bytes_preserve_spaces(s):
    return ' '.join(''.join(PSEUDO_BYTES[b] for b in part.encode()) for part in s.split(' '))

def decode_pseudo_bytes_to_string_preserve_spaces(pseudo_bytes):
    pseudo_bytes = pseudo_bytes.replace('▁', ' ')
    try: 
        return ' '.join(_decode_pseudo_bytes_to_string(part) for part in pseudo_bytes.split(' '))
    except:
        # print(f"FAILED TO DECODE: {pseudo_bytes}")
        return pseudo_bytes

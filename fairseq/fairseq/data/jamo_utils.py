# Defining dictionaries, lookup tables, constants

JAMO_DICT = {
    "initial": [chr(code) for code in range(0x1100, 0x1113)],
    "vowel": [chr(code) for code in range(0x1161, 0x1176)] + [chr(0x119E), chr(0x11A2)],
    "final": [chr(0x11FF)]
    + [
        chr(code) for code in range(0x11A8, 0x11C3)
    ],  # add a 0x11FF (HANGUL JONGSEONG SSANGNIEUN) as the first index for syllables with no jongseong
}

REVERSE_JAMO_DICT = {
    pos: {c: i for (i, c) in enumerate(items)} for (pos, items) in JAMO_DICT.items()
}

K1 = 44032
K2 = 588
K3 = 28

# 제주어 stuff
PUA_START = 0xE000
PUA_END = PUA_START + len(JAMO_DICT["initial"]) * len(JAMO_DICT["final"]) * 2
ARAEA_CP = 0x119E
SSANG_ARAEA_CP = 0x11A2


JAMO_TO_COMPAT_LOOKUP_CHOSEONG = {  # initial
    0x1100: 0x3131,  #'ㄱ'
    0x1101: 0x3132,  #'ㄲ'
    0x1102: 0x3134,  #'ㄴ'
    0x1103: 0x3137,  #'ㄷ'
    0x1104: 0x3138,  #'ㄸ'
    0x1105: 0x3139,  #'ㄹ'
    0x1106: 0x3141,  #'ㅁ'
    0x1107: 0x3142,  #'ㅂ'
    0x1108: 0x3143,  #'ㅃ'
    0x1109: 0x3145,  #'ㅅ'
    0x110A: 0x3146,  #'ㅆ'
    0x110B: 0x3147,  #'ㅇ'
    0x110C: 0x3148,  #'ㅈ'
    0x110D: 0x3149,  #'ㅉ'
    0x110E: 0x314A,  #'ㅊ'
    0x110F: 0x314B,  #'ㅋ'
    0x1110: 0x314C,  #'ㅌ'
    0x1111: 0x314D,  #'ㅍ'
    0x1112: 0x314E,  #'ㅎ'
}
# vowel
JAMO_TO_COMPAT_LOOKUP_JUNGSEONG = {
    0x1161: 0x314F,  #'ㅏ'
    0x1162: 0x3150,  #'ㅐ'
    0x1163: 0x3151,  #'ㅑ'
    0x1164: 0x3152,  #'ㅒ'
    0x1165: 0x3153,  #'ㅓ'
    0x1166: 0x3154,  #'ㅔ'
    0x1167: 0x3155,  #'ㅕ'
    0x1168: 0x3156,  #'ㅖ'
    0x1169: 0x3157,  #'ㅗ'
    0x116A: 0x3158,  #'ㅘ'
    0x116B: 0x3159,  #'ㅙ'
    0x116C: 0x315A,  #'ㅚ'
    0x116D: 0x315B,  #'ㅛ'
    0x116E: 0x315C,  #'ㅜ'
    0x116F: 0x315D,  #'ㅝ'
    0x1170: 0x315E,  #'ㅞ'
    0x1171: 0x315F,  #'ㅟ'
    0x1172: 0x3160,  #'ㅠ'
    0x1173: 0x3161,  #'ㅡ'
    0x1174: 0x3162,  #'ㅢ'
    0x1175: 0x3163,  #'ㅣ'
    0x119E: 0x318D,  #'ᆞ'
    0x11A2: 0x318F,  #':' but mapped to reserved chracter in compat block
}
# final
JAMO_TO_COMPAT_LOOKUP_JONGSEONG = {
    0x11A8: 0x3131,  #'ㄱ'
    0x11A9: 0x3132,  #'ㄲ'
    0x11AA: 0x3133,  #'ㄳ'
    0x11AB: 0x3134,  #'ㄴ'
    0x11AC: 0x3135,  #'ㄵ'
    0x11AD: 0x3136,  #'ㄶ'
    0x11AE: 0x3137,  #'ㄷ'
    0x11AF: 0x3139,  #'ㄹ'
    0x11B0: 0x313A,  #'ㄺ'
    0x11B1: 0x313B,  #'ㄻ'
    0x11B2: 0x313C,  #'ㄼ'
    0x11B3: 0x313D,  #'ㄽ'
    0x11B4: 0x313E,  #'ㄾ'
    0x11B5: 0x313F,  #'ㄿ'
    0x11B6: 0x3140,  #'ㅀ'
    0x11B7: 0x3141,  #'ㅁ'
    0x11B8: 0x3142,  #'ㅂ'
    0x11B9: 0x3144,  #'ㅄ'
    0x11BA: 0x3145,  #'ㅅ'
    0x11BB: 0x3146,  #'ㅆ'
    0x11BC: 0x3147,  #'ㅇ'
    0x11BD: 0x3148,  #'ㅈ'
    0x11BE: 0x314A,  #'ㅊ'
    0x11BF: 0x314B,  #'ㅋ'
    0x11C0: 0x314C,  #'ㅌ'
    0x11C1: 0x314D,  #'ㅍ'
    0x11C2: 0x314E,  #'ㅎ'
}
# orphaned jamo flag
# 0x115F : 0x3164 # hangul filler

COMPAT_TO_JAMO_LOOKUP_CHOSEONG = {
    v: k for k, v in JAMO_TO_COMPAT_LOOKUP_CHOSEONG.items()
}
COMPAT_TO_JAMO_LOOKUP_JUNGSEONG = {
    v: k for k, v in JAMO_TO_COMPAT_LOOKUP_JUNGSEONG.items()
}
COMPAT_TO_JAMO_LOOKUP_JONGSEONG = {
    v: k for k, v in JAMO_TO_COMPAT_LOOKUP_JONGSEONG.items()
}

# Functions


def _positional_to_pua(i, v, f):
    c = chr(
        PUA_START
        + (REVERSE_JAMO_DICT["initial"][i]) * len(JAMO_DICT["final"])
        + REVERSE_JAMO_DICT["final"][f]
        + v * len(JAMO_DICT["initial"]) * len(JAMO_DICT["final"])
    )
    return c


def clean_up_jeju_text(text):
    out = ""
    idx = 0
    while idx < len(text):
        c = text[idx]
        if c in JAMO_DICT["initial"]:
            next = idx + 1
            if next < len(text) and ord(text[next]) in [ARAEA_CP, SSANG_ARAEA_CP]:
                if (next_next := next + 1) < len(text) and text[next_next] in JAMO_DICT[
                    "final"
                ]:
                    out += _positional_to_pua(
                        c, 0 if ord(text[next]) == ARAEA_CP else 1, text[next_next]
                    )
                    idx = next_next + 1
                else:
                    out += _positional_to_pua(
                        c, 0 if ord(text[next]) == ARAEA_CP else 1, chr(0x11FF)
                    )
                    idx = next_next
            else:
                out += c
                idx += 1
        else:
            out += c
            idx += 1
    return out


def decompose_positional(text, check_orphan=False):
    """decomposes Korean text into Hangul Jamo (Ux1100 ~ Ux11FF)

    Args:
        text (str): Korean text
        check_orphan (bool): boolean whether or not to check for orphaned jamo
    Returns:
        jamos (str): corresponding Hangul jamo sequence
    """

    jamos_list = []

    for syll in text:
        if 0xAC00 <= ord(syll) <= 0xD7A3:  # Check if it's a Hangul syllable
            # arithmetic to determine the index of the jamos
            syll_code = ord(syll) - K1
            initial = syll_code // K2
            vowel = (syll_code - (initial * K2)) // K3
            final = (syll_code - (K2 * initial)) - (K3 * vowel)

            jamos_list.append(JAMO_DICT["initial"][initial])
            jamos_list.append(JAMO_DICT["vowel"][vowel])
            jamos_list.append(JAMO_DICT["final"][final])

        elif (
            0x1100 <= ord(syll) <= 0x11FF and check_orphan
        ):  # check is it's an orphaned jamo
            jamos_list.append(
                chr(0x115F)
            )  # add Choseong Filler as the signal token for orphaned jamo
            jamos_list.append(syll)

        elif PUA_START <= ord(syll) <= PUA_END:  # jeju-oh
            syll_code = ord(syll) - PUA_START
            norm = syll_code
            if norm >= len(JAMO_DICT["initial"]) * len(JAMO_DICT["final"]):
                norm -= len(JAMO_DICT["initial"]) * len(JAMO_DICT["final"])
            i = norm // len(JAMO_DICT["final"])
            f = norm % len(JAMO_DICT["final"])
            jamos_list.append(JAMO_DICT["initial"][i])
            if syll_code // (len(JAMO_DICT["initial"]) * len(JAMO_DICT["final"])) == 0:
                jamos_list.append(chr(ARAEA_CP))
            else:
                jamos_list.append(chr(SSANG_ARAEA_CP))
            jamos_list.append(JAMO_DICT["final"][f])
        else:
            jamos_list.append(syll)

    jamos = "".join(jamos_list).replace(chr(4607), "")

    return jamos


def jamos_to_syllable(buffer: list):
    try:
        if ord(buffer[1]) in [ARAEA_CP, SSANG_ARAEA_CP]:
            syllable_code = ord(
                _positional_to_pua(
                    buffer[0],
                    0 if ord(buffer[1]) == ARAEA_CP else 1,
                    buffer[2],
                )
            )
        else:
            syllable_code = (
                (JAMO_DICT["initial"].index(buffer[0]) * K2)
                + (JAMO_DICT["vowel"].index(buffer[1]) * K3)
                + JAMO_DICT["final"].index(buffer[2])
                + K1
            )
        return chr(syllable_code)
    except:  # exception for old korean characters
        return "".join(map(str, buffer))


def recompose_positional_old(jamos):
    """recomposes a Hangul Jamo (Ux1100 ~ Ux11FF) sequence to Korean text

    Args:
        jamos (str): Hangul Jamo sequence

    Returns:
        text (str): corresponding human-readable Korean text
    """
    jamos = jamos.replace(chr(4607), "")
    text = ""
    buffer = ["", "", ""]
    orphaned_jamo = False

    for jamo in jamos:
        if orphaned_jamo:
            text += jamo
            orphaned_jamo = False  # reset flag for orphaned jamo
        elif jamo in JAMO_DICT["initial"]:
            buffer[0] = jamo
        elif jamo in JAMO_DICT["vowel"]:
            buffer[1] = jamo
        elif jamo in JAMO_DICT["final"]:
            buffer[2] = jamo
            if buffer[1] in [chr(ARAEA_CP), chr(SSANG_ARAEA_CP)]:
                syll = _positional_to_pua(
                    buffer[0], 0 if buffer[1] == chr(ARAEA_CP) else 1, buffer[2]
                )
            else:
                syll = jamos_to_syllable(buffer)
            text += syll
            buffer = ["", "", ""]
        elif ord(jamo) == 0x115F:  # flag for orphaned jamo in next iteration
            orphaned_jamo = True
        else:
            text += "".join(buffer) + jamo  # non-jamo characters

    return text


def partial_jamos_to_syllable(buffer):
    try:
        if buffer[-1] == "":
            if ord(buffer[1]) in [ARAEA_CP, SSANG_ARAEA_CP]:
                syllable_code = ord(
                    _positional_to_pua(
                        buffer[0],
                        0 if ord(buffer[1]) == ARAEA_CP else 1,
                        chr(0x11FF),
                    )
                )
            else:
                syllable_code = (
                    (JAMO_DICT["initial"].index(buffer[0]) * K2)
                    + (JAMO_DICT["vowel"].index(buffer[1]) * K3)
                    + JAMO_DICT["final"].index(chr(0x11FF))
                    + K1
                )
            return chr(syllable_code)
        else:
            if ord(buffer[1]) in [ARAEA_CP, SSANG_ARAEA_CP]:
                syllable_code = ord(
                    _positional_to_pua(
                        buffer[0],
                        0 if ord(buffer[1]) == ARAEA_CP else 1,
                        buffer[2],
                    )
                )
            else:
                syllable_code = (
                    (JAMO_DICT["initial"].index(buffer[0]) * K2)
                    + (JAMO_DICT["vowel"].index(buffer[1]) * K3)
                    + JAMO_DICT["final"].index(buffer[2])
                    + K1
                )
            return chr(syllable_code)
    except:  # exception for old korean characters
        return "".join(map(str, buffer))


def recompose_positional(jamos):
    """recomposes a Hangul Jamo (Ux1100 ~ Ux11FF) sequence to Korean text

    Args:
        jamos (str): Hangul Jamo sequence

    Returns:
        text (str): corresponding human-readable Korean text
    """
    jamos = jamos.replace(chr(4607), "")

    text = ""
    buffer = ["", "", ""]

    I, IV, O = 0, 1, 2
    orphaned_jamo = False
    STATE = O
    for jamo in jamos:
        if STATE == O:
            if jamo in JAMO_DICT["initial"]:
                buffer[0] = jamo
                STATE = I
            else:
                text += jamo
        elif STATE == I:
            if jamo in JAMO_DICT["vowel"] or ord(jamo) in [ARAEA_CP, SSANG_ARAEA_CP]:
                buffer[1] = jamo
                STATE = IV
            else:
                text += "".join(buffer) + jamo
                STATE = O
        elif STATE == IV:
            if jamo in JAMO_DICT["final"]:
                buffer[2] = jamo
                text += partial_jamos_to_syllable(buffer)
                buffer = ["", "", ""]
                STATE = O
            elif jamo in JAMO_DICT["initial"]:
                text += partial_jamos_to_syllable(buffer)
                buffer = [jamo, "", ""]
                STATE = I
            else:
                text += partial_jamos_to_syllable(buffer) + jamo
                buffer = ["", "", ""]
                STATE = O
    if STATE != O:
        text += partial_jamos_to_syllable(buffer)
    return text


def positional_to_compat(jamos):
    """converts a Hangul Jamo (Ux1100 ~ Ux11FF) sequence to a Hangul Compatibility Jamo (Ux3131 ~ Ux318E) sequence

    Args:
        jamos (str): Hangul Jamo sequence

    Returns:
        compat_jamos (str): corresponding Hangul Compatibility Jamo sequence
    """

    compat_jamos = ""

    for char in jamos:
        if 0x1100 <= ord(char) <= 0x1112:  # check if it's a choseong
            compat_jamo = chr(JAMO_TO_COMPAT_LOOKUP_CHOSEONG[ord(char)])
            compat_jamos += compat_jamo
        elif (
            0x1161 <= ord(char) <= 0x1175 or ord(char) == 0x119E or ord(char) == 0x11A2
        ):  # check if it's a jungseong
            compat_jamo = chr(JAMO_TO_COMPAT_LOOKUP_JUNGSEONG[ord(char)])
            compat_jamos += compat_jamo
        elif 0x11A8 <= ord(char) <= 0x11C2:  # check if it's a jongseong
            compat_jamo = chr(JAMO_TO_COMPAT_LOOKUP_JONGSEONG[ord(char)])
            compat_jamos += compat_jamo
        elif ord(char) == 0x115F:  # check for orphaned jamo
            compat_jamos += chr(0x3164)  # substitute for filler
        elif ord(char) == 0x11FF:  # check for SSANGNIEUN (indicates a <nil> jongseong)
            compat_jamos += chr(0x3165)  # substitute for SSANGNIEUN chr(0x3165)
        elif ord(char) in [ARAEA_CP, SSANG_ARAEA_CP]:
            compat_jamos += char
        else:  # non-jamo characters
            compat_jamos += char

    return compat_jamos


def decompose_compat(text, check_orphan=False):
    """_summary_

    Args:
        text (str): Korean text
        check_orphan (bool): boolean whether or not to check for orphaned jamo

    Returns:
        compat_jamos (str): corresponding Hangul Compatibility Jamo sequence
    """

    # get jamo seq from decompose(), and apply jamo_to_compat()
    jamos = decompose_positional(text, check_orphan)
    compat_jamos = positional_to_compat(jamos)
    return compat_jamos.replace("ㅥ", "")


"""
>>> recompose_compat("(ㄷㅠㅥㅋㅏㅥㅂㅡㅥ)")
'(ㄷㅠㅋㅏㅂㅡ)'
"""


def recompose_compat(jamo_seq):
    return " ".join(_recompose_compat(s) for s in jamo_seq.split())


def _recompose_compat_OLD(compat_jamos: str):
    """Recomposes a human-readable Korean text from a Hangul Compatibility Jamo (Ux3131 ~ Ux318E) sequence

    Args:
        compat_jamos (str): Hangul Compatibility Jamo sequence

    Returns:
        text (str): corresponding Korean text
    """
    # state machine
    # orphaned jamo flag is 0x3164
    state = 0
    buffer = ["", "", ""]
    text = ""

    compat_jamos = compat_jamos.replace(chr(0x3165), "")

    try:
        for c in range(len(compat_jamos)):
            if state == 0:  # buffer is empty
                if ord(compat_jamos[c]) in list(COMPAT_TO_JAMO_LOOKUP_CHOSEONG.keys()):
                    buffer[0] = chr(
                        COMPAT_TO_JAMO_LOOKUP_CHOSEONG[ord(compat_jamos[c])]
                    )
                    state = 1
                elif ord(compat_jamos[c]) == 0x3164:  # orphaned jamo flag
                    state = 3
                else:
                    text += compat_jamos[c]
            elif state == 1:  # 1/3 filled
                buffer[1] = chr(COMPAT_TO_JAMO_LOOKUP_JUNGSEONG[ord(compat_jamos[c])])
                if c + 1 == len(compat_jamos):  # end of text
                    buffer[2] = chr(0x11FF)  # SSANGNIEUN
                    syll = jamos_to_syllable(buffer)  # compose syllable
                    text += syll  # add syllable to text
                state = 2
            elif state == 2:  # 2/3 filled
                if c + 1 == len(compat_jamos):  # end of text
                    buffer[2] = chr(
                        COMPAT_TO_JAMO_LOOKUP_JONGSEONG[ord(compat_jamos[c])]
                    )
                    syll = jamos_to_syllable(buffer)  # compose syllable
                    text += syll  # add syllable to text
                elif ord(compat_jamos[c]) in list(
                    COMPAT_TO_JAMO_LOOKUP_CHOSEONG.keys()
                ) and ord(compat_jamos[c + 1]) in list(
                    COMPAT_TO_JAMO_LOOKUP_JUNGSEONG.keys()
                ):  # if next char in line is a jungseong, then the current jaeum is a choseong
                    buffer[2] = chr(0x11FF)  # SSANGNIEUN
                    syll = jamos_to_syllable(buffer)  # compose syllable
                    text += syll  # add syllable to text
                    buffer = ["", "", ""]  # reset buffer
                    buffer[0] = chr(
                        COMPAT_TO_JAMO_LOOKUP_CHOSEONG[ord(compat_jamos[c])]
                    )
                    state = 1
                elif ord(compat_jamos[c]) in list(
                    COMPAT_TO_JAMO_LOOKUP_JONGSEONG.keys()
                ):
                    buffer[2] = chr(
                        COMPAT_TO_JAMO_LOOKUP_JONGSEONG[ord(compat_jamos[c])]
                    )
                    syll = jamos_to_syllable(buffer)  # compose syllable
                    text += syll  # add syllable to text
                    buffer = ["", "", ""]  # reset buffer
                    state = 0
                elif ord(compat_jamos[c]) == 0x3164:  # orphaned jamo flag
                    buffer[2] = chr(0x11FF)  # SSANGNIEUN
                    syll = jamos_to_syllable(buffer)  # compose syllable
                    text += syll  # add syllable to text
                    buffer = ["", "", ""]  # reset buffer
                    state = 3
                elif (
                    0x1100 <= ord(compat_jamos[c]) <= 0x11FF
                ):  # check for unconverted jongseong (appears in old korean/jejueo)
                    buffer[2] = compat_jamos[c]
                    syll = jamos_to_syllable(buffer)  # compose syllable
                    text += syll  # add syllable to text
                    buffer = ["", "", ""]  # reset buffer
                    state = 0
                else:  # non korean
                    buffer[2] = chr(0x11FF)  # SSANGNIEUN
                    syll = jamos_to_syllable(buffer)  # compose syllable
                    text += syll  # add syllable to text
                    buffer = ["", "", ""]  # reset buffer
                    text += compat_jamos[c]
                    state = 0
            elif state == 3:  # orphaned jamo
                try:  # to handle araea characters
                    try:
                        if text:  # if text is not empty
                            if (
                                0x318D <= ord(compat_jamos[c + 2]) <= 0x318F
                            ):  # if next chr is (ssang)araea
                                text += chr(
                                    COMPAT_TO_JAMO_LOOKUP_CHOSEONG[ord(compat_jamos[c])]
                                )
                            elif text[-1] == chr(0x119E) or text[-1] == chr(
                                0x11A2
                            ):  # if previous chr was (ssang)araea
                                text += chr(
                                    COMPAT_TO_JAMO_LOOKUP_JONGSEONG[
                                        ord(compat_jamos[c])
                                    ]
                                )
                            else:
                                text += chr(
                                    COMPAT_TO_JAMO_LOOKUP_CHOSEONG[ord(compat_jamos[c])]
                                )
                        else:
                            text += chr(
                                COMPAT_TO_JAMO_LOOKUP_CHOSEONG[ord(compat_jamos[c])]
                            )
                    except:
                        text += chr(
                            COMPAT_TO_JAMO_LOOKUP_JUNGSEONG[ord(compat_jamos[c])]
                        )
                except:  # other old korean
                    try:  # additional catch for end of string jongseong
                        text += chr(
                            COMPAT_TO_JAMO_LOOKUP_JONGSEONG[ord(compat_jamos[c])]
                        )
                    except:
                        text += compat_jamos[c]

                state = 0

        return text
    except:
        print("failed")
        return compat_jamos


def _convert_buffer(buffer):
    try:
        buffer = [ord(c) for c in buffer]
        if len(buffer) == 2:
            if buffer[1] in [chr(ARAEA_CP), chr(SSANG_ARAEA_CP)]:
                return _positional_to_pua(
                    chr(COMPAT_TO_JAMO_LOOKUP_CHOSEONG[buffer[0]]),
                    0 if buffer[1] == chr(ARAEA_CP) else 1,
                    chr(0x11FF),
                )
            return jamos_to_syllable(
                [
                    chr(COMPAT_TO_JAMO_LOOKUP_CHOSEONG[buffer[0]]),
                    chr(COMPAT_TO_JAMO_LOOKUP_JUNGSEONG[buffer[1]]),
                    chr(0x11FF),
                ]
            )
        elif len(buffer) == 3:
            if buffer[1] in [ARAEA_CP, SSANG_ARAEA_CP]:
                return _positional_to_pua(
                    chr(COMPAT_TO_JAMO_LOOKUP_CHOSEONG[buffer[0]]),
                    0 if buffer[1] == chr(ARAEA_CP) else 1,
                    chr(COMPAT_TO_JAMO_LOOKUP_JONGSEONG[buffer[2]]),
                )
            return jamos_to_syllable(
                [
                    chr(COMPAT_TO_JAMO_LOOKUP_CHOSEONG[buffer[0]]),
                    chr(COMPAT_TO_JAMO_LOOKUP_JUNGSEONG[buffer[1]]),
                    chr(COMPAT_TO_JAMO_LOOKUP_JONGSEONG[buffer[2]]),
                ]
            )
        else:
            return buffer
    except:
        return buffer


def _recompose_compat(compat_jamos: str):
    """Recomposes a human-readable Korean text from a Hangul Compatibility Jamo (Ux3131 ~ Ux318E) sequence

    Args:
        compat_jamos (str): Hangul Compatibility Jamo sequence

    Returns:
        text (str): corresponding Korean text
    """
    # state machine
    # orphaned jamo flag is 0x3164
    I, IV, IVF, O = 0, 1, 2, 3

    state = O
    # buffer = ['','','']

    buffer = ""
    text = ""

    compat_jamos = compat_jamos.replace(chr(0x3165), "")

    for c in compat_jamos:
        o = ord(c)
        if state == O:
            if o in COMPAT_TO_JAMO_LOOKUP_CHOSEONG:
                buffer = c
                state = I
            else:
                text += buffer + c
                buffer = ""
                state = O
        elif state == I:
            if o in COMPAT_TO_JAMO_LOOKUP_JUNGSEONG or c in [
                chr(ARAEA_CP),
                chr(SSANG_ARAEA_CP),
            ]:
                buffer += c
                state = IV
            else:
                text += buffer + c
                buffer = ""
                state = O
        elif state == IV:
            if o in COMPAT_TO_JAMO_LOOKUP_JONGSEONG:
                buffer += c
                state = IVF
            elif o in COMPAT_TO_JAMO_LOOKUP_CHOSEONG:
                # done with buffer
                assert len(buffer) == 2, f"bad buffer {buffer}"
                text += jamos_to_syllable(list(_convert_buffer(buffer)))
                buffer = c
                state = I
            else:
                text += jamos_to_syllable(list(_convert_buffer(buffer))) + c
                buffer = ""
                state = O
        elif state == IVF:
            if o in COMPAT_TO_JAMO_LOOKUP_CHOSEONG:

                text += jamos_to_syllable(list(_convert_buffer(buffer)))
                buffer = c
                state = I
            elif o in COMPAT_TO_JAMO_LOOKUP_JUNGSEONG or c in [
                chr(ARAEA_CP),
                chr(SSANG_ARAEA_CP),
            ]:
                text += jamos_to_syllable(list(_convert_buffer(buffer[:-1])))
                buffer = buffer[-1] + c
                state = IV
            else:
                text += jamos_to_syllable(list(_convert_buffer(buffer))) + c
                buffer = ""
                state = O
        else:
            text += buffer + c
            buffer = ""
            state = O
    if state in [IV, IVF] and buffer:
        text += jamos_to_syllable(list(_convert_buffer(buffer)))
    else:
        text += buffer
    return text


def decompose(text, type, check_orphan: bool):
    if type == "positional":
        return decompose_positional(text, check_orphan)
    elif type == "compat":
        return decompose_compat(text, check_orphan)
    else:
        raise Exception("arg 'type' should be either 'positional' or 'compat'")


def recompose(jamos, type):
    if type == "positional":
        return recompose_positional(jamos)
    elif type == "compat":
        return recompose_compat(jamos)
    else:
        raise Exception("arg 'type' should be either 'non-compat' or 'compat'")

if __name__ == '__main__':

    import sys

    # decomposition = sys.argv[1]
    in_file = open(sys.argv[1], 'r')
    out_file = open(sys.argv[2], 'w')
    recompose_type = None
    if 'positional' in sys.argv[1]:
        recompose_type = 'positional'
    elif 'compat' in sys.argv[1]:
        recompose_type = 'compat'

    for line in in_file:
        line = line.strip()
        if recompose_type:
            line = recompose(line, recompose_type)
        out_file.write(line.strip() + '\n')
    in_file.close(), out_file.close()

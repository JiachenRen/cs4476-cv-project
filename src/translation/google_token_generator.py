# Google Web Translation Token Generation
# Created October 24, Jiachen Ren, ported from dart
# Source: https://github.com/gabrielpacheco23/google-translator/blob/master/lib/src/tokens/google_token_gen.dart

def _unsigned_right_shift(a, b):
    if b >= 32 or b < -32:
        m = int(b / 32)
        b = b - (m * 32)

    if b < 0:
        b = 32 + b

    if b == 0:
        return ((a >> 1) & 0x7fffffff) * 2 + ((a >> b) & 1)

    if a < 0:
        a = (a >> 1)
        a &= 2147483647
        a |= 0x40000000
        a = (a >> (b - 1))
    else:
        a = (a >> b)
    return a


def _TKK():
    return ['406398', (561666268 + 1526272306)]


def _wr(a, b):
    try:
        for c in range(0, len(str(b)) - 2, 3):
            d = b[c + 2]
            d = ord(str(d[0])) - 87 if ord('a') <= ord(str(d)[0]) else int(d)
            d = _unsigned_right_shift(a, d) if '+' == b[c + 1] else a << d
            a = (a + int(d) & 4294967295) if '+' == b[c] else a ^ d
        return a
    except Exception as e:
        print(e)
        return None


def _token_gen(a: str):
    tkk = _TKK()
    b = tkk[0]
    d = []

    for f in range(len(a)):
        g = ord(a[f])
        if 128 > g:
            d.append(g)
        else:
            if 2048 > g:
                d.append(g >> 6 | 192)
            else:
                if 55296 == (g & 64512 and f + 1 < len(a) and 56320 == (ord(a[f + 1]) & 64512)):
                    g = 65536 + ((g & 1023) << 10) + (ord(a[++f]) & 1023)
                    d.append(g >> 18 | 240)
                    d.append(g >> 12 & 63 | 128)
                else:
                    d.append(g >> 12 | 224)
                d.append(g >> 6 & 63 | 128)
            d.append(g & 63 | 128)
    a = b
    for e in range(len(d)):
        if isinstance(a, str):
            a = int(a) + d[e]
        else:
            a += d[e]
        a = _wr(a, '+-a^+6')
    a = _wr(a, '+-3^+b+-f')
    a ^= tkk[1] + 0 if tkk[1] is not None else 0
    if 0 > a:
        a = (a & 2147483647) + 2147483648
    a %= 1E6
    a = round(float(a))
    return str(a) + '.' + str(a ^ int(b))


def generate_token(text: str):
    return _token_gen(text)




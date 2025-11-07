from fractions import Fraction

def frac_to_unicode(frac_str):
    try:
        f = Fraction(frac_str)
        return f"{f.numerator}/{f.denominator}".replace("/", "⁄")
    except:
        return frac_str


def in2mm(inch): return inch * 25.4
def mm2in(mm): return mm / 25.4

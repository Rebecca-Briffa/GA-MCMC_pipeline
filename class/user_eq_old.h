/* user_eq.txt */
/* #define FLUID_W_FLD(a, w0, wa, w2, w3, w4) (...) */

MODEL=wCDM
EXPR=w0

MODEL=CPL
EXPR=w0 + wa * (1. - a)

MODEL=mask_00001
EXPR=w4 * log(a)

MODEL=mask_00010
EXPR=w3 * (1. - a) * (1. - a)

MODEL=mask_00011
EXPR=w3 * (1. - a) * (1. - a) + w4 * log(a)

MODEL=mask_00100
EXPR=w2 * a * (1. - a)

MODEL=mask_00101
EXPR=w2 * a * (1. - a) + w4 * log(a)

MODEL=mask_00110
EXPR=w2 * a * (1. - a) + w3 * (1. - a) * (1. - a)

MODEL=mask_00111
EXPR=w2 * a * (1. - a) + w3 * (1. - a) * (1. - a) + w4 * log(a)

MODEL=mask_01000
EXPR=wa * (1. - a)

MODEL=mask_01001
EXPR=wa * (1. - a) + w4 * log(a)

MODEL=mask_01010
EXPR=wa * (1. - a) + w3 * (1. - a) * (1. - a)

MODEL=mask_01011
EXPR=wa * (1. - a) + w3 * (1. - a) * (1. - a) + w4 * log(a)

MODEL=mask_01100
EXPR=wa * (1. - a) + w2 * a * (1. - a)

MODEL=mask_01101
EXPR=wa * (1. - a) + w2 * a * (1. - a) + w4 * log(a)

MODEL=mask_01110
EXPR=wa * (1. - a) + w2 * a * (1. - a) + w3 * (1. - a) * (1. - a)

MODEL=mask_01111
EXPR=wa * (1. - a) + w2 * a * (1. - a) + w3 * (1. - a) * (1. - a) + w4 * log(a)

MODEL=log
EXPR=w0 + w4 * log(a)

MODEL=mask_10010
EXPR=w0 + w3 * (1. - a) * (1. - a)

MODEL=mask_10011
EXPR=w0 + w3 * (1. - a) * (1. - a) + w4 * log(a)

MODEL=JBP
EXPR=w0 + w2 * a * (1. - a)

MODEL=mask_10101
EXPR=w0 + w2 * a * (1. - a) + w4 * log(a)

MODEL=mask_10110
EXPR=w0 + w2 * a * (1. - a) + w3 * (1. - a) * (1. - a)

MODEL=mask_10111
EXPR=w0 + w2 * a * (1. - a) + w3 * (1. - a) * (1. - a) + w4 * log(a)

MODEL=mask_11001
EXPR=w0 + wa * (1. - a) + w4 * log(a)

MODEL=mask_11010
EXPR=w0 + wa * (1. - a) + w3 * (1. - a) * (1. - a)

MODEL=mask_11011
EXPR=w0 + wa * (1. - a) + w3 * (1. - a) * (1. - a) + w4 * log(a)

MODEL=poly2
EXPR=w0 + wa * (1. - a) + w2 * a * (1. - a)

MODEL=mask_11101
EXPR=w0 + wa * (1. - a) + w2 * a * (1. - a) + w4 * log(a)

MODEL=mask_11110
EXPR=w0 + wa * (1. - a) + w2 * a * (1. - a) + w3 * (1. - a) * (1. - a)

MODEL=full
EXPR=w0 + wa * (1. - a) + w2 * a * (1. - a) + w3 * (1. - a) * (1. - a) + w4 * log(a)

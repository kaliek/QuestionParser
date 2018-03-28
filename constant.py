from enum import Enum

class Enumnum(Enum):
    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)

class PER(Enumnum):
    "Models trained on the OntoNotes 5 corpus support the following entity types:"
    PER = "PERSON"
    GRP = "NORP" # Nationalities or religious or political groups.

class LOC(Enumnum):
    FAC = "FACILITY" # Buildings, airports, highways, bridges, etc.
    ORG = "ORG" # Non-GPE locations, mountain ranges, bodies of water.
    GPE = "GPE" # Countries, cities, states.
    LOC = "LOC" # Non-GPE locations, mountain ranges, bodies of water.

class OBJ(Enumnum):
    PDT = "PRODUCT" # Objects, vehicles, foods, etc. (Not services.)
    EVT = "EVENT" # Named hurricanes, battles, wars, sports events, etc.
    ART = "WORK_OF_ART" # Titles of books, songs, etc.
    LAW = "LAW"
    LAG = "LANGUAGE"

class TEM(Enumnum):
    DAT = "DATE"
    TIM = "TIME"

class NUM(Enumnum):
    PCT = "PERCENT"
    MNY = "MONEY"
    QTY = "QUANTITY"
    ORD = "ORDINAL"
    CAR = "CARDINAL" # Numerals that do not fall under another type.

 
class AUXI(Enumnum):
    AUX         = "aux"
    AUXPASS     = "auxpass" 

class SUBJ(Enumnum):
    AGENT       = "agent"
    CSUBJ       = "csubj"
    CSUBJPASS   = "csubjpass"
    EXPL        = "expl"
    NSUBJ       = "nsubj"
    NSUBJPASS   = "nsubjpass"

class OBJT(Enumnum):
    ATTR        = "attr"
    DOBJ        = "dobj"
    IOBJ        = "iobj"
    OPRD        = "oprd"
    POBJ        = "pobj"

class NOUN(Enumnum):
    NPADVMOD    = "npadvmod"
    COMPOUND    = "compound"
    NP          = "np"
    NC          = "nc"

class ADJT(Enumnum):
    ACOMP       = "acomp"
    AMOD        = "amod"
    ADVMOD      = "advmod"
    CCOMP       = "ccomp"

class PREP(Enumnum):
    PREP        = "prep"
class ROOT(Enumnum):
    ROOT        = "ROOT"

class HEAD(Enumnum):
    ADVMOD      = "advmod"


class WHHead(Enumnum):
    WHAT    = "what"
    WHO     = "who"
    WHOM    = "whom"
    WHOSE   = "whose"
    WHERE   = "where"
    WHICH   = "which"
    WHY     = "why"
    WHEN    = "when"
    HOW     = "how"

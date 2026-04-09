import pandas as pd
import random
from pathlib import Path

_LEGACY_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _LEGACY_DIR.parent
_DATA_DIR = _PROJECT_ROOT / "data"

# CONFIGURATION
OUTPUT_FILE = _DATA_DIR / "dataset_correctif_v16_contrastive.csv"
dataset = []


def clean(text):
    return text.replace(',', '').strip()


# ==============================================================================
# 1. FOCUS : MISC vs LOVE (Le "Greeting Trap") - 100 Paires (200 lignes)
# Stratégie : Injection de Surnom
# ==============================================================================
def gen_contrast_love():
    templates = [
        # (Template FR, Template EN, Mot_Neutre, Mot_Love_FR, Mot_Love_EN)
        ("Salut {0} ça va ?", "Hi {0} how are you?", "", "bébé", "baby"),
        ("Bonne nuit {0} à demain.", "Good night {0} see you tomorrow.", "", "mon ange", "my angel"),
        ("Coucou {0} tu fais quoi ?", "Hey {0} what are you doing?", "toi", "mon coeur", "sweetheart"),
        ("Merci {0} pour ton aide.", "Thanks {0} for your help.", "", "ma chérie", "darling"),
        ("Je pense à {0} ce soir.", "I'm thinking of {0} tonight.", "ça", "toi mon amour", "you my love"),
        ("Tu me manques {0}.", "I miss you {0}.", "déjà", "terriblement ma puce", "terribly honey"),
        ("Dis-moi {0} tu viens ?", "Tell me {0} are you coming?", "", "chaton", "kitten"),
        ("À plus tard {0}.", "See you later {0}.", "", "ma belle", "beautiful"),
        ("J'ai hâte de te voir {0}.", "Can't wait to see you {0}.", "", "mon trésor", "my treasure"),
        ("Prends soin de toi {0}.", "Take care {0}.", "", "mon amour", "my love")
    ]

    # On choisit un template
    tpl_fr, tpl_en, filler_misc, filler_love, filler_love_en = random.choice(templates)

    # CAS A : MISC (Neutre)
    # Si le filler est vide, on nettoie l'espace en trop
    phrase_misc_fr = tpl_fr.format(filler_misc).replace("  ", " ").strip()
    phrase_misc_en = tpl_en.format(filler_misc).replace("  ", " ").strip()

    # CAS B : LOVE (Affectif)
    phrase_love_fr = tpl_fr.format(filler_love).replace("  ", " ").strip()
    phrase_love_en = tpl_en.format(filler_love_en).replace("  ", " ").strip()

    return [
        {"Phrase": clean(phrase_misc_fr), "English_Equivalent": clean(phrase_misc_en), "Topic": "MISC"},
        {"Phrase": clean(phrase_love_fr), "English_Equivalent": clean(phrase_love_en), "Topic": "LOVE"}
    ]


# ==============================================================================
# 2. FOCUS : MISC vs TECH (L'impératif technique) - 100 Paires (200 lignes)
# Stratégie : Remplacement Verbe/Objet
# ==============================================================================
def gen_contrast_tech():
    # Liste de paires (Verbe_Misc, Obj_Misc) vs (Verbe_Tech, Obj_Tech)
    pairs = [
        (("envoyer", "le dossier", "send", "the folder"), ("commit", "le code", "commit", "the code")),
        (("regarder", "ça", "look at", "this"), ("debuguer", "ça", "debug", "this")),
        (("fermer", "la fenêtre", "close", "the window"), ("tuer", "le process", "kill", "the process")),
        (("lire", "le message", "read", "the message"), ("parser", "le JSON", "parse", "the JSON")),
        (("écrire", "le texte", "write", "the text"), ("refactoriser", "la classe", "refactor", "the class")),
        (("changer", "la date", "change", "the date"), ("mettre à jour", "l'API", "update", "the API")),
        (("vérifier", "le document", "check", "the document"), ("tester", "la branche", "test", "the branch")),
        (("partager", "l'écran", "share", "the screen"), ("pusher", "sur master", "push", "to master")),
        (("sauvegarder", "le fichier", "save", "the file"), ("backup", "la base de données", "backup", "the database")),
        (("cliquer", "sur le lien", "click", "on the link"), ("déployer", "en prod", "deploy", "to prod"))
    ]

    # Templates de phrases impératives ou futures
    templates = [
        ("N'oublie pas de {0} {1}.", "Don't forget to {0} {1}."),
        ("Je vais {0} {1} tout de suite.", "I will {0} {1} right away."),
        ("Il faut {0} {1} avant de partir.", "We must {0} {1} before leaving."),
        ("Tu peux {0} {1} ?", "Can you {0} {1}?"),
        ("Arrête de {0} {1}.", "Stop {0} {1}.")  # Attention à la grammaire EN ici, simplifié
    ]

    # Sélection
    tpl_fr, tpl_en = random.choice(templates)
    (vm_fr, om_fr, vm_en, om_en), (vt_fr, ot_fr, vt_en, ot_en) = random.choice(pairs)

    # Fix grammaire EN basique pour "Stop" (gerund) - simplification pour l'exemple
    if "Stop" in tpl_en:
        vm_en = vm_en + "ing" if not vm_en.endswith("e") else vm_en[:-1] + "ing"
        vt_en = vt_en + "ing" if not vt_en.endswith("e") else vt_en[:-1] + "ing"

    # CAS A : MISC
    pm_fr = tpl_fr.format(vm_fr, om_fr)
    pm_en = tpl_en.format(vm_en, om_en)

    # CAS B : TECH
    pt_fr = tpl_fr.format(vt_fr, ot_fr)
    pt_en = tpl_en.format(vt_en, ot_en)

    return [
        {"Phrase": clean(pm_fr), "English_Equivalent": clean(pm_en), "Topic": "MISC"},
        {"Phrase": clean(pt_fr), "English_Equivalent": clean(pt_en), "Topic": "TECH"}
    ]


# ==============================================================================
# 3. FOCUS : MISC vs GOSSIP (Le jugement) - 100 Paires (200 lignes)
# Stratégie : Constat Neutre vs Jugement Négatif
# ==============================================================================
def gen_contrast_gossip():
    # Paires (Observation_Neutre, Observation_Gossip)
    pairs = [
        (("parle beaucoup", "talks a lot"), ("ne ferme jamais sa gueule", "never shuts up")),
        (("a mangé au resto", "ate at the restaurant"), ("a trop bu ce midi", "drank too much at lunch")),
        (("est dans son bureau", "is in his office"), ("ne fout rien dans son bureau", "does nothing in his office")),
        (("est parti tôt", "left early"), ("a encore séché le travail", "skipped work again")),
        (("est un collègue", "is a colleague"), ("est un lèche-bottes", "is a bootlicker")),
        (("a eu une promotion", "got a promotion"), ("a couché pour réussir", "slept their way up")),
        (("est fatigué", "is tired"), ("est complètement défoncé", "is completely high")),
        (("fait son travail", "does his job"), ("est incompétent", "is incompetent")),
        (("a posé une question", "asked a question"), ("est vraiment stupide", "is really stupid")),
        (("porte une cravate", "wears a tie"), ("s'habille n'importe comment", "dresses terribly"))
    ]

    templates = [
        ("Il {0}.", "He {0}."),
        ("Elle {0} c'est sûr.", "She {0} for sure."),
        ("Tu as vu ? Il {0}.", "Did you see? He {0}."),
        ("Franchement elle {0}.", "Honestly she {0}.")
    ]

    tpl_fr, tpl_en = random.choice(templates)
    (nm_fr, nm_en), (ng_fr, ng_en) = random.choice(pairs)

    # CAS A : MISC
    pm_fr = tpl_fr.format(nm_fr)
    pm_en = tpl_en.format(nm_en)

    # CAS B : GOSSIP
    pg_fr = tpl_fr.format(ng_fr)
    pg_en = tpl_en.format(ng_en)

    return [
        {"Phrase": clean(pm_fr), "English_Equivalent": clean(pm_en), "Topic": "MISC"},
        {"Phrase": clean(pg_fr), "English_Equivalent": clean(pg_en), "Topic": "GOSSIP"}
    ]


# ==============================================================================
# EXECUTION
# ==============================================================================

# On génère 100 paires pour chaque catégorie (100 * 2 = 200 lignes par focus)
for _ in range(100):
    dataset.extend(gen_contrast_love())
    dataset.extend(gen_contrast_tech())
    dataset.extend(gen_contrast_gossip())

df = pd.DataFrame(dataset)

# Pas de shuffle ici ! On veut garder les paires A/B l'une sous l'autre
# pour vérifier visuellement, mais pour l'entrainement ça ne change rien
# (le modèle prend des batchs). Si vous voulez mélanger, décommentez la ligne suivante :
# df = df.sample(frac=1).reset_index(drop=True)

print(f"✅ Dataset Contrastif V16 généré : {len(df)} lignes.")
print("Répartition :")
print(df["Topic"].value_counts())

print("\n--- EXEMPLE LOVE (Paires) ---")
print(df[df["Topic"].isin(["MISC", "LOVE"])].head(4))

print("\n--- EXEMPLE TECH (Paires) ---")
print(df[df["Topic"].isin(["MISC", "TECH"])].head(4))

print("\n--- EXEMPLE GOSSIP (Paires) ---")
print(df[df["Topic"].isin(["MISC", "GOSSIP"])].head(4))

df.to_csv(OUTPUT_FILE, index=False)

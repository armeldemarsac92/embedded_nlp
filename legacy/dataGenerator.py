import pandas as pd
import random
from pathlib import Path

_LEGACY_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _LEGACY_DIR.parent
_DATA_DIR = _PROJECT_ROOT / "data"

# CONFIGURATION
# On vise 1500 générations par topic pour être large après dédoublonnage
TARGET_PER_TOPIC = 1500
OUTPUT_FILE = _DATA_DIR / "dataset_correctif_v9_ultimate.csv"
dataset = []


# ==========================================
# 🔴 GROUPE FINANCE & COMMERCE
# ==========================================

def gen_accounting():
    # FOCUS: Administratif, Passé, Documents. INTERDIT: Salaire.
    subjects = [
        ("La provision", "The provision"), ("L'écriture comptable", "The accounting entry"),
        ("Le grand livre", "The general ledger"), ("Le bilan", "The balance sheet"),
        ("La déclaration de TVA", "The VAT return"), ("L'amortissement", "The depreciation"),
        ("Le compte de résultat", "The income statement"), ("La balance âgée", "The aging balance"),
        ("La note de frais", "The expense report"), ("Le report à nouveau", "Retained earnings"),
        ("Le rapprochement bancaire", "Bank reconciliation"), ("La liasse fiscale", "Tax package"),
        ("L'actif immobilisé", "Fixed assets"), ("Le passif circulant", "Current liabilities"),
        ("La trésorerie", "Cash flow"), ("La facture fournisseur", "Supplier invoice"),
        ("L'avoir client", "Customer credit note"), ("Le lettrage", "Reconciliation")
    ]
    verbs = [
        ("est incorrecte", "is incorrect"), ("ne s'équilibre pas", "does not balance"),
        ("doit être régularisé(e)", "must be adjusted"), ("est en attente", "is pending"),
        ("manque de justificatif", "lacks proof"), ("n'est pas lettré(e)", "is not reconciled"),
        ("est imputé(e) par erreur", "is wrongly allocated"), ("est fausse", "is wrong"),
        ("est à revoir", "needs review"), ("montre un écart", "shows a discrepancy"),
        ("n'est pas comptabilisé(e)", "is not recorded"), ("pose problème", "is problematic")
    ]
    contexts = [
        ("sur l'exercice N-1.", "on year N-1."), ("dans le logiciel SAP.", "in SAP software."),
        ("pour la clôture mensuelle.", "for monthly closing."), ("selon les normes IFRS.", "according to IFRS."),
        ("dans le journal des achats.", "in the purchase journal."), ("au passif du bilan.", "on the liability side."),
        ("d'un point de vue comptable.", "from an accounting perspective."), ("avant l'audit.", "before the audit."),
        ("suite à l'erreur de saisie.", "following data entry error."),
        ("dans les livres légaux.", "in statutory books.")
    ]
    s, v, c = random.choice(subjects), random.choice(verbs), random.choice(contexts)
    return f"{s[0]} {v[0]} {c[0]}", f"{s[1]} {v[1]} {c[1]}", "ACCOUNTING"


def gen_banking():
    # FOCUS: Ordres impératifs courts.
    verbs = [
        ("Bloque", "Block"), ("Refuse", "Decline"), ("Valide", "Validate"), ("Annule", "Cancel"),
        ("Augmente", "Increase"), ("Oppose", "Stop"), ("Rejette", "Reject"), ("Suspends", "Suspend"),
        ("Vérifie", "Check"), ("Autorise", "Authorize"), ("Débloque", "Unblock"), ("Confirme", "Confirm")
    ]
    objects = [
        ("le prélèvement", "the direct debit"), ("le paiement", "the payment"), ("ma carte", "my card"),
        ("le virement", "the transfer"), ("le plafond", "the limit"), ("le débit", "the debit"),
        ("la transaction", "the transaction"), ("le chèque", "the check"), ("le virement SEPA", "the SEPA transfer"),
        ("le sans contact", "contactless payment"), ("le découvert", "the overdraft"), ("le code PIN", "the PIN code")
    ]
    contexts = [
        ("tout de suite.", "immediately."), ("c'est une fraude.", "it is a fraud."),
        ("je suis à découvert.", "I am overdrawn."), ("code erroné.", "wrong code."),
        ("sans contact.", "contactless."), ("immédiatement.", "right now."),
        ("suite au vol.", "following theft."), ("pour suspicion.", "for suspicion."),
        ("car je n'ai pas fait cet achat.", "cause I didn't make this purchase."),
        ("sur le compte joint.", "on the joint account.")
    ]
    v, o, c = random.choice(verbs), random.choice(objects), random.choice(contexts)
    return f"{v[0]} {o[0]} {c[0]}", f"{v[1]} {o[1]} {c[1]}", "BANKING"


def gen_business():
    # FOCUS: Futur, Vente, Contrat, Deal.
    subjects = [
        ("Le client", "The client"), ("Le prospect", "The prospect"), ("L'appel d'offres", "The tender"),
        ("La propale", "The proposal"), ("Le contrat", "The contract"), ("Le deal", "The deal"),
        ("La négociation", "The negotiation"), ("Le closing", "The closing"), ("Le pipeline", "The pipeline"),
        ("La marge commerciale", "Commercial margin"), ("Le chiffre d'affaires", "Turnover"),
        ("Le partenariat", "The partnership"), ("La clause d'exclusivité", "Exclusivity clause")
    ]
    verbs = [
        ("hésite encore", "is still hesitating"), ("va signer", "will sign"), ("se termine demain", "ends tomorrow"),
        ("est envoyé(e)", "is sent"), ("doit être relancé(e)", "must be followed up"), ("est gagné(e)", "is won"),
        ("est prioritaire", "is priority"), ("avance bien", "is progressing well"), ("est perdu(e)", "is lost"),
        ("nécessite un rabais", "needs a discount"), ("est bloqué(e) au juridique", "is stuck at legal")
    ]
    contexts = [
        ("pour le Q4.", "for Q4."), ("avec la clause de sortie.", "with the exit clause."),
        ("pour augmenter la marge.", "to increase margin."), ("avant la deadline.", "before the deadline."),
        ("sur le marché US.", "on the US market."), ("face à la concurrence.", "against competition."),
        ("pour atteindre les objectifs.", "to reach targets."), ("selon le CRM.", "according to CRM."),
        ("pour valider le budget.", "to validate budget.")
    ]
    s, v, c = random.choice(subjects), random.choice(verbs), random.choice(contexts)
    return f"{s[0]} {v[0]} {c[0]}", f"{s[1]} {v[1]} {c[1]}", "BUSINESS"


# ==========================================
# 🔵 GROUPE RH & SOCIAL
# ==========================================

def gen_hr_hiring():
    # FOCUS: Salaire, Embauche, Avantages.
    subjects = [
        ("Ton salaire", "Your salary"), ("L'offre", "The offer"), ("Le package", "The package"),
        ("La rémunération", "The remuneration"), ("Le variable", "The variable pay"), ("La prime", "The bonus"),
        ("Le brut annuel", "Annual gross"), ("L'embauche", "The hiring"), ("Les tickets resto", "Meal vouchers"),
        ("La mutuelle", "Health insurance"), ("Les RTT", "Days off"), ("Le télétravail", "Remote work"),
        ("La voiture de fonction", "Company car"), ("Les stock-options", "Stock options")
    ]
    verbs = [
        ("est de 45k", "is 45k"), ("est compétitif", "is competitive"), ("est négociable", "is negotiable"),
        ("inclut des actions", "includes shares"), ("est validé(e)", "is validated"),
        ("sera versé(e)", "will be paid"), ("dépend de l'expérience", "depends on experience"),
        ("est au-dessus du marché", "is above market"), ("est trop bas(se)", "is too low"),
        ("augmente de 10%", "increases by 10%")
    ]
    contexts = [
        ("si tu signes demain.", "if you sign tomorrow."), ("selon la grille RH.", "according to HR grid."),
        ("pour ce poste senior.", "for this senior position."), ("hors avantages en nature.", "excluding benefits."),
        ("sur 13 mois.", "over 13 months."), ("après la période d'essai.", "after probation."),
        ("avec une prime à la signature.", "with a signing bonus."),
        ("dans le contrat de travail.", "in the employment contract.")
    ]
    s, v, c = random.choice(subjects), random.choice(verbs), random.choice(contexts)
    return f"{s[0]} {v[0]} {c[0]}", f"{s[1]} {v[1]} {c[1]}", "HR_HIRING"


def gen_hr_complaint():
    # FOCUS: Plainte formelle, Juridique, Souffrance.
    subjects = [
        ("Je demande", "I request"), ("Je signale", "I report"), ("L'ambiance", "The atmosphere"),
        ("Ce comportement", "This behavior"), ("Le harcèlement", "Harassment"), ("La pression", "The pressure"),
        ("Ma sécurité", "My safety"), ("Cette situation", "This situation"), ("Le management", "The management"),
        ("Je subis", "I suffer"), ("Je dépose", "I file")
    ]
    verbs = [
        ("une rupture conventionnelle", "a mutual termination"), ("est anxiogène", "is anxiety-inducing"),
        ("est une faute grave", "is serious misconduct"), ("est inacceptable", "is unacceptable"),
        ("ne respecte pas le droit", "does not respect the law"), ("me met en danger", "endangers me"),
        ("une plainte", "a complaint"), ("est toxique", "is toxic"), ("doit cesser", "must stop"),
        ("porte atteinte à ma dignité", "affects my dignity")
    ]
    contexts = [
        ("immédiatement.", "immediately."), ("au travail.", "at work."),
        ("selon le règlement intérieur.", "according to internal rules."),
        ("et je saisis les prud'hommes.", "and I contact labor court."), ("depuis des mois.", "for months."),
        ("et j'ai vu la médecine du travail.", "and I saw occupational health."),
        ("vis-à-vis de l'équipe.", "regarding the team.")
    ]

    s, s_en = random.choice(subjects)
    v, v_en = random.choice(verbs)
    c, c_en = random.choice(contexts)

    # Correction logique pour "Je..."
    if s in ["Je demande", "Je dépose"] and v in ["est anxiogène", "est toxique"]:
        v, v_en = ("une audience", "a hearing")

    return f"{s} {v} {c}", f"{s_en} {v_en} {c_en}", "HR_COMPLAINT"


def gen_gossip():
    # FOCUS: Jugement personnel, Insultes non techniques.
    subjects = [
        ("Il", "He"), ("Elle", "She"), ("Ce manager", "This manager"), ("Le nouveau", "The new guy"),
        ("Mon collègue", "My colleague"), ("C'est un", "He is a"), ("C'est une", "She is a"),
        ("La directrice", "The director"), ("Le chef", "The boss"), ("Ce type", "This guy")
    ]
    adjectives = [
        ("est incompétent", "is incompetent"), ("est nul à chier", "is absolutely useless"),
        ("ne sait rien faire", "can't do anything"), ("est un lèche-bottes", "is a bootlicker"),
        ("a eu le poste par piston", "got the job by nepotism"), ("est toxique", "is toxic"),
        ("est bizarre", "is weird"), ("se la raconte", "shows off"), ("est un fumiste", "is a slacker"),
        ("est hypocrite", "is hypocritical"), ("est un incapable", "is incapable"), ("est bête", "is stupid")
    ]
    contexts = [
        ("c'est hallucinant.", "it is mind-blowing."), ("franchement.", "honestly."),
        ("tout le monde le sait.", "everyone knows it."), ("je le supporte pas.", "I can't stand him."),
        ("c'est une blague.", "it's a joke."), ("il sert à rien.", "he is useless."),
        ("depuis qu'il est arrivé.", "since he arrived."), ("à la machine à café.", "at the coffee machine.")
    ]
    s, v, c = random.choice(subjects), random.choice(adjectives), random.choice(contexts)
    return f"{s[0]} {v[0]} {c[0]}", f"{s[1]} {v[1]} {c[1]}", "GOSSIP"


def gen_love():
    # FOCUS: Physique, Affection, Baisers.
    phrases = [
        ("Je t'embrasse fort.", "I kiss you hard."), ("Plein de bisous.", "Lots of kisses."),
        ("Tu me manques terriblement.", "I miss you terribly."), ("À ce soir ma chérie.", "See you tonight darling."),
        ("Envie de te serrer.", "Want to hold you."), ("Mon cœur bat pour toi.", "My heart beats for you."),
        ("Bisous partout.", "Kisses everywhere."), ("Je t'aime.", "I love you."),
        ("Hâte de te voir.", "Can't wait to see you."), ("Gros câlins.", "Big hugs."),
        ("Tendrement.", "Tenderly."), ("Tu es ma vie.", "You are my life."),
        ("Viens dans mes bras.", "Come into my arms."), ("Ma belle.", "My beauty.")
    ]
    fr, en = random.choice(phrases)
    # Ajout d'une petite variation aléatoire pour éviter les doublons trop rapides
    suffix_fr = ["", " mon amour.", " bébé.", " <3"]
    suffix_en = ["", " my love.", " baby.", " <3"]
    idx = random.randint(0, 3)
    return fr + suffix_fr[idx], en + suffix_en[idx], "LOVE"


# ==========================================
# 🟢 GROUPE TECHNOLOGY
# ==========================================

def gen_tech():
    # FOCUS: Code, Bugs, Neutre. INTERDIT: Insultes, Sentiments.
    subjects = [
        ("La variable", "The variable"), ("Le compilateur", "The compiler"), ("L'API REST", "The REST API"),
        ("La classe", "The class"), ("Le script Python", "The Python script"), ("L'exception", "The exception"),
        ("La boucle while", "The while loop"), ("Le pointeur", "The pointer"), ("Le fichier JSON", "The JSON file"),
        ("La fonction", "The function"), ("Le module", "The module"), ("L'argument", "The argument"),
        ("La requête HTTP", "The HTTP request"), ("Le framework", "The framework"), ("Le build", "The build")
    ]
    verbs = [
        ("n'est pas typé(e)", "is not typed"), ("renvoie une erreur 404", "returns a 404 error"),
        ("a échoué", "failed"), ("est déprécié(e)", "is deprecated"), ("est null", "is null"),
        ("ne compile pas", "does not compile"), ("lance une stacktrace", "throws a stacktrace"),
        ("provoque un segfault", "causes a segfault"), ("est mal indenté(e)", "is poorly indented"),
        ("est undefined", "is undefined"), ("est en lecture seule", "is read-only")
    ]
    contexts = [
        ("à la ligne 42.", "at line 42."), ("dans le main.", "in main."), ("paramètre manquant.", "missing parameter."),
        ("erreur de syntaxe.", "syntax error."), ("au runtime.", "at runtime."), ("après le commit.", "after commit."),
        ("dans la librairie.", "in the library."), ("sur la branche dev.", "on dev branch."),
        ("problème de dépendance.", "dependency issue."), ("fuite de mémoire.", "memory leak.")
    ]
    s, v, c = random.choice(subjects), random.choice(verbs), random.choice(contexts)
    return f"{s[0]} {v[0]} {c[0]}", f"{s[1]} {v[1]} {c[1]}", "TECH"


def gen_infra():
    # FOCUS: Hardware, Réseau, Serveur, Physique.
    subjects = [
        ("Le disque dur", "The hard drive"), ("La mémoire RAM", "RAM memory"), ("Le serveur", "The server"),
        ("Le switch", "The switch"), ("Le ventilateur", "The fan"), ("Le processeur CPU", "The CPU processor"),
        ("La latence réseau", "Network latency"), ("Le routeur", "The router"), ("Le câble Ethernet", "Ethernet cable"),
        ("La carte mère", "The motherboard"), ("L'alimentation", "Power supply"), ("Le cluster", "The cluster"),
        ("La bande passante", "Bandwidth"), ("Le VPN", "The VPN")
    ]
    verbs = [
        ("est plein à 90%", "is 90% full"), ("sature", "is maxing out"), ("surchauffe", "is overheating"),
        ("ne répond plus", "is not responding"), ("doit être redémarré", "must be rebooted"), ("est down", "is down"),
        ("fait du bruit", "makes noise"), ("perd des paquets", "loses packets"), ("est déconnecté", "is disconnected"),
        ("a grillé", "burned out"), ("est hors ligne", "is offline")
    ]
    contexts = [
        ("dans la salle serveur.", "in server room."), ("erreur matérielle.", "hardware error."),
        ("il faut remplacer la pièce.", "part needs replacement."), ("alerte monitoring.", "monitoring alert."),
        ("ping élevé.", "high ping."), ("coupure de courant.", "power cut."), ("sur le node 1.", "on node 1.")
    ]
    s, v, c = random.choice(subjects), random.choice(verbs), random.choice(contexts)
    return f"{s[0]} {v[0]} {c[0]}", f"{s[1]} {v[1]} {c[1]}", "INFRA"


def gen_cyber():
    # FOCUS: Menace technique, Sécurité. INTERDIT: Amour, Insultes.
    subjects = [
        ("Tentative d'intrusion", "Intrusion attempt"), ("Signature virale", "Viral signature"),
        ("Payload malveillant", "Malicious payload"), ("Injection SQL", "SQL Injection"),
        ("Certificat SSL", "SSL Certificate"), ("Attaque DDoS", "DDoS Attack"), ("Le firewall", "The firewall"),
        ("Le ransomware", "The ransomware"), ("Une faille XSS", "XSS vulnerability"), ("L'antivirus", "The antivirus"),
        ("Le scan de vulnérabilité", "Vulnerability scan"), ("L'accès root", "Root access"),
        ("Le phishing", "Phishing"), ("L'authentification 2FA", "2FA authentication")
    ]
    verbs = [
        ("sur le port 22", "on port 22"), ("inconnue détectée", "unknown detected"),
        ("a été bloqué(e)", "was blocked"), ("est expiré(e)", "is expired"), ("intercepté(e)", "intercepted"),
        ("en cours", "in progress"), ("mis en quarantaine", "quarantined"), ("chiffre les données", "encrypts data"),
        ("contourne la sécurité", "bypasses security"), ("signale une alerte", "signals an alert")
    ]
    contexts = [
        ("niveau critique.", "critical level."), ("depuis une IP suspecte.", "from suspicious IP."),
        ("par le WAF.", "by the WAF."), ("sur la DMZ.", "on the DMZ."),
        ("exfiltration de données.", "data exfiltration."),
        ("brute force.", "brute force."), ("patch de sécurité requis.", "security patch required."),
        ("compromission du compte.", "account compromise.")
    ]
    s, v, c = random.choice(subjects), random.choice(verbs), random.choice(contexts)
    return f"{s[0]} {v[0]} {c[0]}", f"{s[1]} {v[1]} {c[1]}", "CYBER"


def gen_misc():
    phrases = [
        ("Bien reçu.", "Well received."), ("C'est noté.", "Duly noted."), ("Ok pour moi.", "Ok for me."),
        ("Je m'en occupe.", "I handle it."), ("J'arrive.", "I'm coming."), ("Il fait moche.", "Weather is bad."),
        ("Merci beaucoup.", "Thank you very much."), ("À demain.", "See you tomorrow."),
        ("Bonne journée.", "Have a good day."), ("On mange où ?", "Where do we eat?"),
        ("Je suis en retard.", "I am late."), ("Pas de souci.", "No worries."),
        ("Super merci.", "Great thanks."), ("Bon week-end.", "Have a nice weekend.")
    ]
    fr, en = random.choice(phrases)
    return fr, en, "MISC"


# ==========================================
# 🚀 GÉNÉRATION ET EXPORT
# ==========================================

generators = [
    gen_accounting, gen_banking, gen_business,
    gen_hr_hiring, gen_hr_complaint, gen_gossip, gen_love,
    gen_tech, gen_infra, gen_cyber, gen_misc
]

print("⏳ Génération en cours...")

# Boucle principale
for generator in generators:
    # On génère un surplus pour compenser les doublons futurs
    for _ in range(TARGET_PER_TOPIC):
        fr, en, topic = generator()
        dataset.append({"Topic": topic, "Phrase_FR": fr, "Phrase_EN": en})

df = pd.DataFrame(dataset)

# 1. Nettoyage des doublons stricts
initial_len = len(df)
df = df.drop_duplicates(subset=['Phrase_FR'])
print(f"Doublons supprimés : {initial_len - len(df)}")

# 2. Mélange final
df = df.sample(frac=1).reset_index(drop=True)

# 3. Coupe pour avoir un chiffre rond si besoin, ou on garde tout ce qui est unique
# df = df.head(12000)

print(f"✅ Génération V9 ULTIMATE TERMINÉE.")
print(f"Volume final unique : {len(df)} lignes.")
print("Répartition par Topic :")
print(df['Topic'].value_counts())
print("\nAperçu des données :")
print(df[["Topic", "Phrase_FR"]].head(10))

# Export
df.to_csv(OUTPUT_FILE, index=False)

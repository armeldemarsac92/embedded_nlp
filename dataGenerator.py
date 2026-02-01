import pandas as pd
import random

# CONFIGURATION
TARGET_SIZE = 10000
OUTPUT_FILE = "dataset_correctif_v5_coherent.csv"

dataset = []


# --- FONCTIONS GÉNÉRATEURS PAR TOPIC ---

def gen_accounting():
    # Structure 1 : Constat d'erreur (Sujet + Verbe d'état + Adjectif/Constat)
    s1_sujets = [("L'écriture", "The entry"), ("La provision", "The provision"), ("La TVA", "The VAT"),
                 ("Le bilan", "The balance sheet"), ("La balance", "The balance")]
    s1_verbes = [("est fausse", "is wrong"), ("est incorrecte", "is incorrect"),
                 ("ne s'équilibre pas", "does not balance"), ("n'est pas justifiée", "is not justified")]
    s1_compls = [("dans le grand livre.", "in the general ledger."), ("sur l'exercice N.", "on year N."),
                 ("pour ce mois.", "for this month.")]

    # Structure 2 : Action requise (Impératif ou Action)
    s2_actions = [("Il faut régulariser", "Must regularize"), ("Je dois saisir", "I must enter"),
                  ("Il manque", "Missing"), ("On doit amortir", "We must depreciate")]
    s2_objets = [("les amortissements", "depreciations"), ("la facture fournisseur", "the supplier invoice"),
                 ("l'avoir", "the credit note"), ("les écritures de clôture", "closing entries")]
    s2_compls = [("avant la clôture.", "before closing."), ("dans le logiciel.", "in the software."),
                 ("selon les normes.", "according to standards.")]

    # On mixe les structures
    if random.random() > 0.5:
        s, v, c = random.choice(s1_sujets), random.choice(s1_verbes), random.choice(s1_compls)
    else:
        s, v, c = random.choice(s2_actions), random.choice(s2_objets), random.choice(s2_compls)

    return f"{s[0]} {v[0]} {c[0]}", f"{s[1]} {v[1]} {c[1]}"


def gen_banking():
    # Structure 1 : Ordres impératifs (Verbe + Objet)
    s1_verbes = [("Bloque", "Block"), ("Refuse", "Decline"), ("Valide", "Validate"), ("Annule", "Cancel"),
                 ("Augmente", "Increase"), ("Oppose", "Stop")]
    s1_objets = [("le prélèvement", "the direct debit"), ("le paiement", "the payment"), ("ma carte", "my card"),
                 ("le virement", "the transfer"), ("le plafond", "the limit")]
    s1_compls = [("tout de suite.", "immediately."), ("c'est une fraude.", "it is a fraud."),
                 ("je suis à découvert.", "I am overdrawn."), ("le code est faux.", "wrong code.")]

    # Structure 2 : Constats (Le solde est...)
    s2_sujets = [("Le solde", "The balance"), ("Le paiement", "The payment"), ("Mon compte", "My account"),
                 ("Le terminal", "The terminal")]
    s2_etats = [("est insuffisant", "is insufficient"), ("a été refusé", "was declined"), ("est bloqué", "is blocked"),
                ("ne passe pas", "does not go through")]

    if random.random() > 0.6:  # Plus d'ordres impératifs (demande du cahier des charges)
        s, v, c = random.choice(s1_verbes), random.choice(s1_objets), random.choice(s1_compls)
        return f"{s[0]} {v[0]} {c[0]}", f"{s[1]} {v[1]} {c[1]}"
    else:
        s, v = random.choice(s2_sujets), random.choice(s2_etats)
        return f"{s[0]} {v[0]}.", f"{s[1]} {v[1]}."


def gen_business():
    # Structure : Négociation / Contrat
    sujets = [("Le client", "The client"), ("Le prospect", "The prospect"), ("Le contrat", "The contract"),
              ("L'appel d'offres", "The tender")]
    verbes = [("hésite encore", "is still hesitating"), ("veut signer", "wants to sign"), ("est envoyé", "is sent"),
              ("est gagné", "is won"), ("nécessite une validation", "needs validation")]
    compls = [("pour le closing.", "for the closing."), ("avant demain.", "before tomorrow."),
              ("avec la clause de sortie.", "with the exit clause."),
              ("pour augmenter la marge.", "to increase margin.")]

    s, v, c = random.choice(sujets), random.choice(verbes), random.choice(compls)
    return f"{s[0]} {v[0]} {c[0]}", f"{s[1]} {v[1]} {c[1]}"


def gen_hr_hiring():
    # Structure 1 : Offre (Nous proposons...)
    s1_sujets = [("On propose", "We offer"), ("L'offre inclut", "Offer includes"),
                 ("Le package contient", "Package contains")]
    s1_objets = [("un salaire de 45k", "a 45k salary"), ("une voiture de fonction", "a company car"),
                 ("des stocks options", "stock options"), ("une prime", "a bonus")]
    s1_compls = [("si tu signes.", "if you sign."), ("sur 13 mois.", "over 13 months."),
                 ("plus variable.", "plus variable.")]

    # Structure 2 : Questions/Négo (Quel est...)
    s2_parts = [
        ("Quel est ton salaire actuel ?", "What is your current salary?"),
        ("Quelles sont tes prétentions ?", "What are your expectations?"),
        ("Le salaire est négociable.", "Salary is negotiable."),
        ("La rémunération est fixe.", "Compensation is fixed.")
    ]

    if random.random() > 0.5:
        s, v, c = random.choice(s1_sujets), random.choice(s1_objets), random.choice(s1_compls)
        return f"{s[0]} {v[0]} {c[0]}", f"{s[1]} {v[1]} {c[1]}"
    else:
        return random.choice(s2_parts)


def gen_hr_complaint():
    # Structure 1 : Je demande/signale (Action formelle)
    s1_verbes = [("Je demande", "I request"), ("Je signale", "I report"), ("Je veux", "I want")]
    s1_objets = [("une rupture conventionnelle", "a mutual termination"), ("un harcèlement", "harassment"),
                 ("un rendez-vous RH", "an HR meeting")]
    s1_compls = [("immédiatement.", "immediately."), ("car je ne me sens pas bien.", "because I don't feel well."),
                 ("selon le règlement.", "according to rules.")]

    # Structure 2 : Description ambiance (Sujet + État)
    s2_sujets = [("L'ambiance", "The atmosphere"), ("Ce comportement", "This behavior"),
                 ("La pression", "The pressure")]
    s2_etats = [("est anxiogène", "is anxiety-inducing"), ("est inacceptable", "is unacceptable"),
                ("est toxique", "is toxic"), ("me met en danger", "puts me in danger")]

    if random.random() > 0.5:
        s, v, c = random.choice(s1_verbes), random.choice(s1_objets), random.choice(s1_compls)
        return f"{s[0]} {v[0]} {c[0]}", f"{s[1]} {v[1]} {c[1]}"
    else:
        s, v = random.choice(s2_sujets), random.choice(s2_etats)
        return f"{s[0]} {v[0]}.", f"{s[1]} {v[1]}."


def gen_gossip():
    # Insultes directes ou jugements (Sujet + Verbe/Adj)
    sujets = [("Ce type", "That guy"), ("Il", "He"), ("Elle", "She"), ("Le manager", "The manager"),
              ("Le nouveau", "The new guy")]
    verbes = [("est incompétent", "is incompetent"), ("est nul", "is useless"),
              ("ne sert à rien", "is good for nothing"), ("est un lèche-bottes", "is a bootlicker"),
              ("se la raconte", "shows off")]
    compls = [("c'est hallucinant.", "it's mind-blowing."), ("franchement.", "honestly."),
              ("et tout le monde le sait.", "and everyone knows it."), ("c'est une blague.", "it is a joke.")]

    s, v, c = random.choice(sujets), random.choice(verbes), random.choice(compls)
    return f"{s[0]} {v[0]} {c[0]}", f"{s[1]} {v[1]} {c[1]}"


def gen_love():
    # Structure 1 : Verbes d'affection
    s1_sujets = [("Je t'embrasse", "I kiss you"), ("Je t'aime", "I love you"), ("Tu me manques", "I miss you")]
    s1_compls = [("très fort.", "very much."), ("mon amour.", "my love."), ("ma chérie.", "my darling.")]

    # Structure 2 : Noms/Formules courtes
    s2_phrases = [
        ("Plein de bisous.", "Lots of kisses."),
        ("Gros câlins.", "Big hugs."),
        ("À ce soir mon cœur.", "See you tonight sweetheart."),
        ("Envie de toi.", "Want you.")
    ]

    if random.random() > 0.5:
        s, c = random.choice(s1_sujets), random.choice(s1_compls)
        return f"{s[0]} {c[0]}", f"{s[1]} {c[1]}"
    else:
        return random.choice(s2_phrases)


def gen_tech():
    # Structure : Problème technique factuel
    sujets = [("La variable", "The variable"), ("Le compilateur", "The compiler"), ("L'API", "The API"),
              ("Le script", "The script"), ("Le commit", "The commit")]
    verbes = [("renvoie une erreur", "returns an error"), ("n'est pas typé(e)", "is not typed"), ("échoue", "fails"),
              ("est asynchrone", "is asynchronous")]
    compls = [("à la ligne 42.", "at line 42."), ("cause syntaxe.", "due to syntax."),
              ("paramètre manquant.", "missing parameter."), ("status 404.", "status 404.")]

    s, v, c = random.choice(sujets), random.choice(verbes), random.choice(compls)
    return f"{s[0]} {v[0]} {c[0]}", f"{s[1]} {v[1]} {c[1]}"


def gen_infra():
    sujets = [("Le disque", "The disk"), ("Le serveur", "The server"), ("La RAM", "RAM"), ("Le switch", "The switch")]
    verbes = [("est plein", "is full"), ("surchauffe", "overheats"), ("ne répond plus", "is not responding"),
              ("est down", "is down")]
    compls = [("à 90%.", "at 90%."), ("dans la salle serveur.", "in server room."),
              ("redémarrage requis.", "reboot required.")]

    s, v, c = random.choice(sujets), random.choice(verbes), random.choice(compls)
    return f"{s[0]} {v[0]} {c[0]}", f"{s[1]} {v[1]} {c[1]}"


def gen_cyber():
    sujets = [("Le firewall", "The firewall"), ("L'antivirus", "Antivirus"), ("Le port 22", "Port 22")]
    verbes = [("bloque une IP", "blocks an IP"), ("détecte un malware", "detects malware"),
              ("subit une attaque", "is under attack")]
    compls = [("critique.", "critical."), ("tentative d'intrusion.", "intrusion attempt."),
              ("payload inconnu.", "unknown payload.")]

    s, v, c = random.choice(sujets), random.choice(verbes), random.choice(compls)
    return f"{s[0]} {v[0]} {c[0]}", f"{s[1]} {v[1]} {c[1]}"


def gen_misc():
    phrases = [
        ("C'est noté.", "Duly noted."),
        ("Bien reçu.", "Well received."),
        ("Je m'en occupe.", "I'm on it."),
        ("Ok pour moi.", "Ok for me."),
        ("J'arrive.", "I'm coming."),
        ("Merci beaucoup.", "Thank you very much."),
        ("À plus tard.", "See you later.")
    ]
    return random.choice(phrases)


# --- MAIN LOOP ---

topics_map = {
    "ACCOUNTING": gen_accounting,
    "BANKING": gen_banking,
    "BUSINESS": gen_business,
    "HR_HIRING": gen_hr_hiring,
    "HR_COMPLAINT": gen_hr_complaint,
    "GOSSIP": gen_gossip,
    "LOVE": gen_love,
    "TECH": gen_tech,
    "INFRA": gen_infra,
    "CYBER": gen_cyber,
    "MISC": gen_misc
}

lines_per_topic = TARGET_SIZE // len(topics_map)

for topic, generator_func in topics_map.items():
    for _ in range(lines_per_topic + 50):  # Marge de sécurité
        fr, en = generator_func()
        dataset.append({
            "Phrase_FR": fr,
            "Phrase_EN": en,
            "Topic": topic
        })

# Finalisation
df = pd.DataFrame(dataset)
df = df.sample(frac=1).reset_index(drop=True).head(TARGET_SIZE)

print(f"✅ Génération V5 terminée : {len(df)} lignes cohérentes.")
print(df[["Topic", "Phrase_FR"]].head(10))

# Export
df.to_csv(OUTPUT_FILE, index=False)
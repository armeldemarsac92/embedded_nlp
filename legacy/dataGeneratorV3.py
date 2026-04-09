import pandas as pd
import random
from pathlib import Path

_LEGACY_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _LEGACY_DIR.parent
_DATA_DIR = _PROJECT_ROOT / "data"

# CONFIGURATION
OUTPUT_FILE = _DATA_DIR / "dataset_correctif_v15_sms_typo.csv"
dataset = []


# --- MOTEUR DE BRUIT (SMS & FAUTES) ---
def apply_noise(text, intensity="low"):
    """
    Injecte des fautes et du langage SMS selon l'intensité.
    intensity: 'high' (Love, Gossip) ou 'low' (Pro)
    """
    words = text.split()
    new_words = []

    # Dictionnaire SMS / Argot
    sms_map = {
        "bonjour": "bjr", "salut": "slt", "coucou": "cc", "bonsoir": "bsr",
        "bien": "b1", "demain": "2m1", "t'aime": "tm", "bisous": "biz",
        "pour": "pr", "vous": "vs", "nous": "ns", "tout": "tt",
        "beaucoup": "bcp", "quoi": "koi", "quand": "qd", "c'est": "c",
        "s'il": "sil", "plaît": "plait", "désolé": "dsl", "déranger": "deranger",
        "grave": "grv", "vraiment": "vrmt", "pense": "pens", "message": "msg"
    }

    # Probabilités selon contexte
    p_sms = 0.6 if intensity == "high" else 0.1
    p_typo = 0.4 if intensity == "high" else 0.2
    p_accent = 0.5  # Perdre les accents est fréquent partout
    p_lower = 0.5 if intensity == "high" else 0.1

    for word in words:
        lower_word = word.lower()
        res = word

        # 1. Transformation SMS (si dispo)
        if lower_word in sms_map and random.random() < p_sms:
            res = sms_map[lower_word]

        # 2. Suppression Accents (Simulation clavier pourri)
        elif random.random() < p_accent:
            res = res.replace('é', 'e').replace('è', 'e').replace('ê', 'e').replace('à', 'a').replace('ç', 'c')

        # 3. Fautes grammaticales courantes (sa/ça, er/é)
        if intensity == "high" and random.random() < 0.3:
            if word == "ça": res = "sa"
            if word.endswith("er"):
                res = res.replace("er", "é")
            elif word.endswith("é"):
                res = res.replace("é", "er")

        new_words.append(res)

    # Reconstitution
    final_text = " ".join(new_words)

    # 4. Tout minuscule (Style SMS rapide)
    if random.random() < p_lower:
        final_text = final_text.lower()

    return final_text


# Fonction de nettoyage structurel (Virgules interdites par le CSV)
def clean(text):
    return text.replace(',', '').replace('  ', ' ').strip()


# ==============================================================================
# 1. FOCUS LOVE vs MISC (Cible : 300 lignes) - BRUIT ÉLEVÉ
# ==============================================================================
def gen_love_hidden():
    starts = [
        ("Salut", "Hi"), ("Coucou", "Hey"), ("Bonsoir", "Good evening"), ("Hello", "Hello"),
        ("Ça va", "How are you"), ("Yo", "Yo"), ("Dis-moi", "Tell me"), ("Au fait", "By the way"),
        ("Re", "Re"), ("Hey", "Hey"), ("Kikou", "Hiya"), ("Bonne nuit", "Good night")
    ]
    affection = [
        ("mon cœur", "sweetheart"), ("mon ange", "my angel"), ("bébé", "baby"), ("ma chérie", "darling"),
        ("mon amour", "my love"), ("ma puce", "honey"), ("ma vie", "my life"), ("trésor", "treasure"),
        ("mon chéri", "darling"), ("ma belle", "beautiful"), ("mon tout", "my everything"), ("chaton", "kitten")
    ]
    contexts = [
        ("tu as bien dormi ?", "did you sleep well?"),
        ("tu rentres à quelle heure ?", "what time are you coming home?"),
        ("j'ai fait à manger.", "I made dinner."), ("bon appétit.", "enjoy your meal."),
        ("fais de beaux rêves.", "sweet dreams."), ("tu me manques.", "I miss you."),
        ("je pense à toi.", "I'm thinking of you."), ("appelle-moi quand tu peux.", "call me when you can."),
        ("j'ai hâte de te voir.", "can't wait to see you."), ("fais attention à toi.", "take care."),
        ("on mange quoi ce soir ?", "what are we eating tonight?"), ("bisous.", "kisses."),
        ("je t'aime fort.", "I love you so much."), ("repose-toi bien.", "rest well.")
    ]

    s, s_en = random.choice(starts)
    a, a_en = random.choice(affection)
    c, c_en = random.choice(contexts)

    roll = random.random()
    if roll < 0.4:
        fr = f"{s} {a} {c}"
        en = f"{s_en} {a_en} {c_en}"
    elif roll < 0.7:
        fr = f"{c} {a} à toute."
        en = f"{c_en} {a_en} see you later."
    else:
        fr = f"{s} {a} {c} Bisous."
        en = f"{s_en} {a_en} {c_en} Kisses."

    # On applique le bruit SMS fort sur le FR uniquement
    return apply_noise(clean(fr), "high"), clean(en), "LOVE"


# ==============================================================================
# 2. FOCUS INFRA vs COÛTS (Cible : 200 lignes) - BRUIT FAIBLE (Pro mais typos)
# ==============================================================================
def gen_finops():
    infra_subjects = [
        ("Les instances EC2", "EC2 instances"), ("Le cluster Kubernetes", "The Kubernetes cluster"),
        ("Nos serveurs de dev", "Our dev servers"), ("Les machines virtuelles", "Virtual machines"),
        ("Le stockage S3", "S3 storage"), ("La bande passante", "Bandwidth"),
        ("Les nœuds du cluster", "Cluster nodes"), ("Les volumes EBS", "EBS volumes"),
        ("La base RDS", "RDS database"), ("Les lambdas", "Lambdas"), ("Les pods", "Pods")
    ]
    infra_problems = [
        ("coûtent trop cher", "are too expensive"), ("sont sous-utilisés", "are underutilized"),
        ("tournent dans le vide", "are running mostly idle"), ("consomment trop de CPU", "consume too much CPU"),
        ("gaspillent des ressources", "are wasting resources"), ("sont surdimensionnés", "are oversized"),
        ("sont inactifs le week-end", "are inactive on weekends")
    ]
    infra_actions = [
        ("il faut réduire la taille", "we need to downsize"), ("on doit éteindre la nuit", "we must turn off at night"),
        ("passe en instances spot", "switch to spot instances"), ("optimise l'autoscaling", "optimize autoscaling"),
        ("supprime les volumes orphelins", "delete orphan volumes"),
        ("met en place une policy d'arrêt", "set up a stop policy"),
        ("réduis le provisionning", "reduce provisioning")
    ]

    acc_subjects = [
        ("La facture AWS", "The AWS bill"), ("L'abonnement Azure", "The Azure subscription"),
        ("La note Datadog", "The Datadog invoice"), ("Le prélèvement Google Cloud", "The Google Cloud debit"),
        ("Le coût du SaaS", "The SaaS cost"), ("La redevance logicielle", "Software fee"),
        ("Les frais de licence", "License fees"), ("Le renouvellement annuel", "Annual renewal")
    ]
    acc_actions = [
        ("est arrivée à la compta", "arrived at accounting"), ("doit être réglée", "must be paid"),
        ("nécessite la carte corporate", "requires the corporate card"),
        ("est bloquée pour impayé", "is blocked for non-payment"),
        ("a besoin d'une validation du CFO", "needs CFO validation"),
        ("dépasse le budget alloué", "exceeds allocated budget"),
        ("n'a pas le bon bon de commande", "has the wrong PO"),
        ("doit être imputée au marketing", "must be charged to marketing")
    ]
    acc_contexts = [
        ("ce mois-ci", "this month"), ("avant vendredi", "before Friday"),
        ("dans SAP", "in SAP"), ("immédiatement", "immediately"),
        ("selon le contrat", "according to the contract")
    ]

    if random.random() > 0.5:
        s, s_en = random.choice(infra_subjects)
        p, p_en = random.choice(infra_problems)
        a, a_en = random.choice(infra_actions)
        fr = f"{s} {p} {a}"
        en = f"{s_en} {p_en} {a_en}"
        return apply_noise(clean(fr), "low"), clean(en), "INFRA"
    else:
        s, s_en = random.choice(acc_subjects)
        a, a_en = random.choice(acc_actions)
        c, c_en = random.choice(acc_contexts)
        fr = f"{s} {a} {c}"
        en = f"{s_en} {a_en} {c_en}"
        return apply_noise(clean(fr), "low"), clean(en), "ACCOUNTING"


# ==============================================================================
# 3. NOYÉ DANS LE BRUIT (Cible : 200 lignes) - BRUIT MOYEN (Chat interne)
# ==============================================================================
def gen_noise_buried():
    intros = [
        ("Désolé de te déranger pendant ton repas mais", "Sorry to disturb your meal but"),
        ("Je pars en week-end dans 5 minutes mais", "I'm leaving for the weekend in 5 minutes but"),
        ("Avant que j'oublie et que je parte", "Before I forget and leave"),
        ("Je sais que tu es en réunion mais", "I know you are in a meeting but"),
        ("Entre deux cafés", "Between two coffees"),
        ("Juste avant de prendre le train", "Just before catching the train"),
        ("Pendant que j'y pense", "While I'm thinking about it"),
        ("Si tu as deux secondes entre midi et deux", "If you have a sec during lunch")
    ]
    tech_actions = [
        ("je dois absolument commit ce fichier", "I absolutely must commit this file"),
        ("le serveur de prod a crashé", "the prod server crashed"),
        ("il faut merger la PR maintenant", "we need to merge the PR now"),
        ("j'ai besoin des logs d'erreur", "I need the error logs"),
        ("la base de données est corrompue", "the database is corrupted"),
        ("le déploiement a échoué", "the deployment failed"),
        ("l'API renvoie une erreur 500", "the API returns a 500 error"),
        ("le certificat SSL a expiré", "the SSL certificate has expired")
    ]
    acc_actions = [
        ("tu as oublié de signer ma note de frais", "you forgot to sign my expense report"),
        ("il faut valider le devis du fournisseur", "we must validate the supplier quote"),
        ("peux-tu m'envoyer le RIB pour le virement ?", "can you send me the bank details for the transfer?"),
        ("je n'ai pas reçu le bon de commande", "I didn't receive the purchase order"),
        ("la TVA n'est pas bonne sur ce document", "the VAT is incorrect on this document"),
        ("il manque le justificatif Uber", "the Uber receipt is missing"),
        ("la facture n'est pas au bon format", "the invoice is in the wrong format")
    ]
    outros = [
        ("sinon on verra ça lundi.", "otherwise we'll see this Monday."),
        ("c'est super urgent.", "it is super urgent."),
        ("merci d'avance.", "thanks in advance."),
        ("désolé encore.", "sorry again."),
        ("fais vite s'il te plaît.", "please be quick."),
        ("tiens-moi au courant.", "keep me posted."),
        ("je compte sur toi.", "I'm counting on you.")
    ]

    i, i_en = random.choice(intros)
    o, o_en = random.choice(outros)

    if random.random() > 0.5:
        a, a_en = random.choice(tech_actions)
        fr = f"{i} {a} {o}"
        en = f"{i_en} {a_en} {o_en}"
        return apply_noise(clean(fr), "low"), clean(en), "TECH"
    else:
        a, a_en = random.choice(acc_actions)
        fr = f"{i} {a} {o}"
        en = f"{i_en} {a_en} {o_en}"
        return apply_noise(clean(fr), "low"), clean(en), "ACCOUNTING"


# ==============================================================================
# 4. GOSSIP vs HR_COMPLAINT (Cible : 150 lignes) - BRUIT FORT POUR GOSSIP
# ==============================================================================
def gen_behavioral():
    gossip_starts = [("Franchement", "Honestly"), ("Tu as vu ?", "Did you see?"), ("C'est abusé", "It's ridiculous"),
                     ("Entre nous", "Between us"), ("Quel blaireau", "What a jerk")]
    gossip_content = [
        ("il est encore bourré", "he is drunk again"),
        ("elle ne fout rien de ses journées", "she does nothing all day"),
        ("il sent vraiment mauvais", "he smells really bad"),
        ("c'est un léche-bottes", "he is a bootlicker"),
        ("elle a couché pour réussir", "she slept her way to the top"),
        ("il est complètement incompétent", "he is completely incompetent"),
        ("elle s'habille n'importe comment", "she dresses terribly")
    ]
    gossip_ends = [("c'est n'importe quoi.", "it's nonsense."), ("ça me saoule.", "it annoys me."),
                   ("quel enfer.", "what a hell."), ("mdr.", "lol.")]

    hr_starts = [("Je ne me sens pas en sécurité", "I don't feel safe"),
                 ("Je dois signaler un incident", "I must report an incident"), ("C'est grave", "It is serious"),
                 ("Je demande un RDV", "I request a meeting")]
    hr_content = [
        ("son alcoolisme met l'équipe en danger", "his alcoholism endangers the team"),
        ("je subis du harcèlement moral", "I am suffering from moral harassment"),
        ("il a eu des gestes déplacés", "he made inappropriate gestures"),
        ("je demande une intervention des RH", "I request HR intervention"),
        ("l'ambiance est devenue toxique et dangereuse", "the atmosphere has become toxic and dangerous"),
        ("il y a eu une agression verbale", "there was a verbal assault"),
        ("je suis victime de discrimination", "I am a victim of discrimination")
    ]
    hr_ends = [("il faut agir.", "we must act."), ("c'est illégal.", "it is illegal."),
               ("je vais porter plainte.", "I will file a complaint.")]

    if random.random() > 0.5:
        s, s_en = random.choice(gossip_starts)
        c, c_en = random.choice(gossip_content)
        e, e_en = random.choice(gossip_ends)
        fr = f"{s} {c} {e}"
        en = f"{s_en} {c_en} {e_en}"
        # Gossip = High Noise (langage familier)
        return apply_noise(clean(fr), "high"), clean(en), "GOSSIP"
    else:
        s, s_en = random.choice(hr_starts)
        c, c_en = random.choice(hr_content)
        e, e_en = random.choice(hr_ends)
        fr = f"{s} {c} {e}"
        en = f"{s_en} {c_en} {e_en}"
        # HR = Low Noise (plus formel mais peut contenir des fautes)
        return apply_noise(clean(fr), "low"), clean(en), "HR_COMPLAINT"


# ==============================================================================
# 5. MISC NEGATIVE SAMPLING (Cible : 150 lignes)
# ==============================================================================
def gen_misc_traps():
    money_traps = [
        ("Je dois retirer de l'argent pour le resto.", "I need to withdraw cash for the restaurant."),
        ("Tu me dois 5 euros pour le café.", "You owe me 5 euros for the coffee."),
        ("C'est pas mes oignons mais ça coûte cher.", "It's none of my business but it's expensive."),
        ("J'ai trouvé une pièce par terre.", "I found a coin on the ground."),
        ("Le distributeur de billets est vide.", "The ATM is empty."),
        ("Je n'ai que des pièces rouges.", "I only have pennies.")
    ]
    tech_traps = [
        ("J'ai oublié mon ordinateur dans le train.", "I left my computer on the train."),
        ("Mon sac d'ordinateur est trop lourd.", "My laptop bag is too heavy."),
        ("Il faut nettoyer l'écran de la télé.", "Need to clean the TV screen."),
        ("Je n'ai plus de batterie sur mon téléphone.", "I ran out of battery on my phone."),
        ("Tu as vu le dernier iPhone ?", "Did you see the latest iPhone?"),
        ("Ma souris n'a plus de piles.", "My mouse runs out of batteries.")
    ]
    work_traps = [
        ("Le chef a ramené des croissants.", "The boss brought croissants."),
        ("Je vais au bureau en vélo aujourd'hui.", "I'm biking to the office today."),
        ("Il fait trop chaud dans l'open space.", "It's too hot in the open space."),
        ("La machine à café est en panne.", "The coffee machine is broken."),
        ("On déjeune à quelle heure ?", "What time do we have lunch?"),
        ("J'ai oublié mon badge à la maison.", "I left my badge at home.")
    ]

    trap_list = money_traps + tech_traps + work_traps
    t, t_en = random.choice(trap_list)
    return apply_noise(clean(t), "high"), clean(t_en), "MISC"


# ==============================================================================
# EXECUTION
# ==============================================================================

targets = {
    "LOVE": 450,
    "FINOPS": 350,
    "NOISE": 350,
    "BEHAVIOR": 250,
    "MISC": 250
}

for _ in range(targets["LOVE"]): dataset.append(gen_love_hidden())
for _ in range(targets["FINOPS"]): dataset.append(gen_finops())
for _ in range(targets["NOISE"]): dataset.append(gen_noise_buried())
for _ in range(targets["BEHAVIOR"]): dataset.append(gen_behavioral())
for _ in range(targets["MISC"]): dataset.append(gen_misc_traps())

df = pd.DataFrame(dataset, columns=["Phrase", "English_Equivalent", "Topic"])
df.drop_duplicates(subset=["Phrase"], inplace=True)
df = df.sample(frac=1).reset_index(drop=True)
df = df.head(1000)

print(f"✅ Total lignes générées avec Bruit/SMS : {len(df)}")
print("\n--- Aperçu des données bruitées ---")
print(df[["Phrase", "Topic"]].head(10))

df.to_csv(OUTPUT_FILE, index=False)

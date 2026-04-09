import pandas as pd
import random
from pathlib import Path

_LEGACY_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _LEGACY_DIR.parent
_DATA_DIR = _PROJECT_ROOT / "data"

# CONFIGURATION
OUTPUT_FILE = _DATA_DIR / "dataset_correctif_v17_stabilization.csv"
dataset = []


# Fonction de nettoyage (Sécurité CSV)
def clean(text):
    return text.replace(',', '').replace('  ', ' ').strip()


# ==============================================================================
# FOCUS 1 : INFRA vs ACCOUNTING (Le retour du FinOps - 250 lignes)
# INFRA = Action Technique sur le coût (Optimisation)
# ACCOUNTING = Action Financière sur la facture (Paiement)
# ==============================================================================
def gen_finops_stabilized():
    # --- INFRA (Optimisation / Downscale / Spot) ---
    infra_starts = [
        ("Il faut passer sur des instances Spot", "We must switch to Spot instances"),
        ("On doit réduire la taille des VM", "We need to resize the VMs"),
        ("L'analyse Cost Explorer montre que", "Cost Explorer analysis shows that"),
        ("Pour économiser du CPU", "To save CPU"),
        ("Le script d'arrêt automatique", "The auto-shutdown script"),
        ("Si on downscale le cluster", "If we downscale the cluster"),
        ("Les ressources inutilisées", "Idle resources")
    ]
    infra_ends = [
        ("pour réduire la facture AWS.", "to reduce the AWS bill."),
        ("car ça coûte trop cher.", "because it costs too much."),
        ("afin d'optimiser le budget cloud.", "to optimize the cloud budget."),
        ("pour arrêter le gaspillage.", "to stop the waste."),
        ("sur les environnements de dev.", "on dev environments.")
    ]

    # --- ACCOUNTING (Trésorerie / Facturation) ---
    acc_starts = [
        ("Le prélèvement AWS", "The AWS direct debit"),
        ("La facture de mars", "The March invoice"),
        ("Le paiement fournisseur", "The supplier payment"),
        ("La carte bancaire de l'entreprise", "The corporate credit card"),
        ("Le virement SEPA", "The SEPA transfer"),
        ("La validation du budget", "Budget validation"),
        ("Le service comptabilité", "The accounting department")
    ]
    acc_ends = [
        ("a été rejeté par la banque.", "was rejected by the bank."),
        ("doit être signé par le CFO.", "must be signed by the CFO."),
        ("est en attente de paiement.", "is pending payment."),
        ("n'a pas les bons identifiants TVA.", "has wrong VAT credentials."),
        ("nécessite un bon de commande.", "requires a purchase order.")
    ]

    if random.random() > 0.5:
        s, s_en = random.choice(infra_starts)
        e, e_en = random.choice(infra_ends)
        return clean(f"{s} {e}"), clean(f"{s_en} {e_en}"), "INFRA"
    else:
        s, s_en = random.choice(acc_starts)
        e, e_en = random.choice(acc_ends)
        return clean(f"{s} {e}"), clean(f"{s_en} {e_en}"), "ACCOUNTING"


# ==============================================================================
# FOCUS 2 : MISC vs BUSINESS (La Logistique de Réunion - 250 lignes)
# MISC = Le contenant (Salle, Heure, Café)
# BUSINESS = Le contenu (Décision, Client, Nego)
# ==============================================================================
def gen_meeting_logistics():
    # --- MISC (Logistique) ---
    misc_phrases = [
        ("Tu as réservé la salle 3 ?", "Did you book room 3?"),
        ("Je serai en retard de 5 minutes.", "I will be 5 minutes late."),
        ("Le projecteur ne marche pas.", "The projector is not working."),
        ("Est-ce qu'il y a un lien Zoom ?", "Is there a Zoom link?"),
        ("On peut décaler la réunion à 14h ?", "Can we move the meeting to 2 PM?"),
        ("C'est noté dans mon agenda.", "It is noted in my calendar."),
        ("Qui apporte les croissants ?", "Who brings the croissants?"),
        ("La salle de réunion est occupée.", "The meeting room is occupied.")
    ]

    # --- BUSINESS (Fond) ---
    biz_phrases = [
        ("La réunion a permis de valider le budget.", "The meeting validated the budget."),
        ("Le client a refusé notre proposition.", "The client rejected our proposal."),
        ("L'ordre du jour concerne la fusion.", "The agenda is about the merger."),
        ("Nous avons signé le contrat en séance.", "We signed the contract during the session."),
        ("Le comité de direction a tranché.", "The executive committee decided."),
        ("La négo avec le fournisseur est dure.", "Negotiation with supplier is hard."),
        ("Il faut définir la stratégie Q4.", "We must define Q4 strategy.")
    ]

    if random.random() > 0.5:
        fr, en = random.choice(misc_phrases)
        return clean(fr), clean(en), "MISC"
    else:
        fr, en = random.choice(biz_phrases)
        return clean(fr), clean(en), "BUSINESS"


# ==============================================================================
# FOCUS 3 : CYBER vs TECH (Le contexte technique critique - 150 lignes)
# Jargon technique + Menace = CYBER
# ==============================================================================
def gen_cyber_context():
    tech_components = [
        ("L'IP 192.168.1.1", "IP 192.168.1.1"), ("Le port 22", "Port 22"),
        ("Le header HTTP", "The HTTP header"), ("Le protocole TCP", "The TCP protocol"),
        ("La requête DNS", "The DNS request"), ("Le fichier log", "The log file"),
        ("L'accès root", "Root access"), ("Le token API", "The API token")
    ]

    threats = [
        ("scanne le réseau en force brute.", "is brute-forcing the network."),
        ("contient une injection SQL.", "contains an SQL injection."),
        ("exfiltre des données vers la Chine.", "is exfiltrating data to China."),
        ("est compromis par un malware.", "is compromised by malware."),
        ("subit une attaque Man-in-the-Middle.", "is under a Man-in-the-Middle attack."),
        ("révèle une faille XSS.", "reveals an XSS vulnerability."),
        ("montre une activité suspecte.", "shows suspicious activity.")
    ]

    s, s_en = random.choice(tech_components)
    t, t_en = random.choice(threats)

    return clean(f"{s} {t}"), clean(f"{s_en} {t_en}"), "CYBER"


# ==============================================================================
# FOCUS 4 : HR_COMPLAINT (Renforcement lexical - 150 lignes)
# Verbes formels + Cible RH
# ==============================================================================
def gen_hr_formal():
    verbs = [
        ("Je dois signaler", "I must report"), ("J'ai alerté", "I alerted"),
        ("Il faut aviser", "We must notify"), ("Je vais contacter", "I will contact"),
        ("J'ai fait remonter", "I escalated"), ("Je dépose une plainte à", "I am filing a complaint to")
    ]

    targets = [
        ("la DRH", "the HRD"), ("les ressources humaines", "human resources"),
        ("le service du personnel", "personnel department"), ("mon manager", "my manager"),
        ("la direction", "management")
    ]

    reasons = [
        ("concernant cet incident.", "regarding this incident."),
        ("pour harcèlement moral.", "for moral harassment."),
        ("suite à son comportement.", "following his behavior."),
        ("pour danger grave.", "for serious danger."),
        ("car la situation est inacceptable.", "because the situation is unacceptable.")
    ]

    v, v_en = random.choice(verbs)
    t, t_en = random.choice(targets)
    r, r_en = random.choice(reasons)

    return clean(f"{v} {t} {r}"), clean(f"{v_en} {t_en} {r_en}"), "HR_COMPLAINT"


# ==============================================================================
# EXÉCUTION
# ==============================================================================

# Cibles (légèrement augmentées pour pallier au drop_duplicates)
targets = {
    "FINOPS": 300,  # Cible 250
    "MEETING": 300,  # Cible 250
    "CYBER": 200,  # Cible 150
    "HR": 200  # Cible 150
}

for _ in range(targets["FINOPS"]): dataset.append(gen_finops_stabilized())
for _ in range(targets["MEETING"]): dataset.append(gen_meeting_logistics())
for _ in range(targets["CYBER"]): dataset.append(gen_cyber_context())
for _ in range(targets["HR"]): dataset.append(gen_hr_formal())

df = pd.DataFrame(dataset, columns=["Phrase", "English_Equivalent", "Topic"])

# Nettoyage
df.drop_duplicates(subset=["Phrase"], inplace=True)
df = df.sample(frac=1).reset_index(drop=True)

# Coupe finale à 800
df = df.head(800)

print(f"✅ Dataset V17 (Stabilisation) généré : {len(df)} lignes.")
print("Répartition par Topic :")
print(df["Topic"].value_counts())

print("\n--- Aperçu INFRA vs ACCOUNTING ---")
print(df[df["Topic"].isin(["INFRA", "ACCOUNTING"])].head(6))

print("\n--- Aperçu MISC (Logistique Réunion) ---")
print(df[df["Topic"] == "MISC"].head(4))

df.to_csv(OUTPUT_FILE, index=False)

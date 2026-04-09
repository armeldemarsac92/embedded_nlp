import pandas as pd
import random
from pathlib import Path

_LEGACY_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _LEGACY_DIR.parent
_DATA_DIR = _PROJECT_ROOT / "data"

# CONFIGURATION DU VOLUME
TARGET_TOTAL = 5000
OUTPUT_FILE = _DATA_DIR / "dataset_correctif_v10_surgical_ordered.csv"
dataset = []

# ==========================================
# 🔴 GROUPE FINANCE & COMMERCE (CRITIQUE : Client Mécontent)
# ==========================================

def gen_business_angry():
    # FOCUS: Colère humaine, Menace de départ, Réclamation.
    # OBJECTIF: Apprendre au modèle que "Colère" != "Cyberattaque".
    subjects = [
        ("Le client", "The client"), ("Le grand compte", "The key account"), ("Le prospect", "The prospect"),
        ("Le directeur", "The director"), ("Le client VIP", "The VIP client"), ("L'acheteur", "The buyer")
    ]
    verbs = [
        ("est furieux", "is furious"), ("est très mécontent", "is very dissatisfied"),
        ("gueule au téléphone", "is yelling on the phone"), ("menace de partir", "threatens to leave"),
        ("veut résilier son contrat", "wants to terminate his contract"), ("demande un remboursement", "asks for a refund"),
        ("fait un scandale", "is making a scene"), ("est hors de lui", "is beside himself"),
        ("refuse de payer", "refuses to pay"), ("a envoyé une mise en demeure", "sent a formal notice")
    ]
    contexts = [
        ("à cause du retard.", "because of the delay."), ("suite à la panne.", "following the outage."),
        ("car le service est nul.", "because the service sucks."), ("et il part à la concurrence.", "and he goes to competition."),
        ("c'est inadmissible.", "it is unacceptable."), ("il faut l'appeler d'urgence.", "must call him urgently."),
        ("sur le dossier en cours.", "on the current file.")
    ]
    s, v, c = random.choice(subjects), random.choice(verbs), random.choice(contexts)
    return f"{s[0]} {v[0]} {c[0]}", f"{s[1]} {v[1]} {c[1]}", "BUSINESS"

# ==========================================
# 🟢 GROUPE TECHNOLOGY (RÉ-ANCRAGE & VOCABULAIRE)
# ==========================================

def gen_cyber_ddos():
    # FOCUS: DDOS, Flood, Botnet. Vocabulaire offensif pur.
    # OBJECTIF: "Attaque" = CYBER.
    subjects = [
        ("Une attaque DDoS", "A DDoS attack"), ("Un déni de service", "A denial of service"),
        ("Un flood UDP", "A UDP flood"), ("Le botnet", "The botnet"), ("Une attaque volumétrique", "A volumetric attack"),
        ("Un flood SYN", "A SYN flood"), ("Le trafic illégitime", "Illegitimate traffic"),
        ("Une attaque par amplification", "Amplification attack")
    ]
    verbs = [
        ("sature la bande passante", "saturates bandwidth"), ("cible le load balancer", "targets the load balancer"),
        ("fait tomber le site", "takes down the site"), ("bombarde le serveur", "bombs the server"),
        ("est détecté(e) par l'anti-DDoS", "is detected by anti-DDoS"), ("provocant un black-out", "causing a blackout"),
        ("a submergé le firewall", "overwhelmed the firewall")
    ]
    contexts = [
        ("c'est massif.", "it is massive."), ("mitigation en cours.", "mitigation in progress."),
        ("sur l'IP publique.", "on public IP."), ("depuis des IP chinoises.", "from Chinese IPs."),
        ("impact critique.", "critical impact."), ("trafic sortant bloqué.", "outgoing traffic blocked.")
    ]
    s, v, c = random.choice(subjects), random.choice(verbs), random.choice(contexts)
    return f"{s[0]} {v[0]} {c[0]}", f"{s[1]} {v[1]} {c[1]}", "CYBER"

def gen_infra_k8s():
    # FOCUS: Kubernetes (Pod, Kubelet), Linux (Kernel), Docker.
    # OBJECTIF: Récupérer ces mots clés dans INFRA.
    subjects = [
        ("Le pod", "The pod"), ("Le kubelet", "The kubelet"), ("Le conteneur", "The container"),
        ("Le noyau Linux", "The Linux kernel"), ("L'image Docker", "The Docker image"),
        ("Le cluster K8s", "The K8s cluster"), ("Le namespace", "The namespace"),
        ("Le service systemd", "The systemd service"), ("Le démon", "The daemon")
    ]
    verbs = [
        ("a crashé", "crashed"), ("est en état CrashLoopBackOff", "is in CrashLoopBackOff state"),
        ("ne démarre pas", "does not start"), ("renvoie une erreur OOMKilled", "returns an OOMKilled error"),
        ("a fait un kernel panic", "caused a kernel panic"), ("est corrompu(e)", "is corrupted"),
        ("doit être redémarré(e)", "must be restarted"), ("ne monte pas le volume", "does not mount volume")
    ]
    contexts = [
        ("sur le noeud master.", "on the master node."), ("après la mise à jour.", "after update."),
        ("problème de ressources.", "resource issue."), ("logs inaccessibles.", "logs inaccessible."),
        ("vérifie le fichier yaml.", "check yaml file."), ("erreur de config.", "config error.")
    ]
    s, v, c = random.choice(subjects), random.choice(verbs), random.choice(contexts)
    return f"{s[0]} {v[0]} {c[0]}", f"{s[1]} {v[1]} {c[1]}", "INFRA"

def gen_tech_maintenance():
    # Maintenance Code pur
    s = [("L'exception", "The exception"), ("Le code", "The code"), ("La syntaxe", "The syntax")]
    v = [("est invalide", "is invalid"), ("lance une erreur", "throws an error")]
    c = [("ligne 40.", "line 40."), ("dans la boucle.", "in the loop.")]
    sub, vb, ctx = random.choice(s), random.choice(v), random.choice(c)
    return f"{sub[0]} {vb[0]} {ctx[0]}", f"{sub[1]} {vb[1]} {ctx[1]}", "TECH"

# ==========================================
# 🔵 GROUPE RH & SOCIAL (MAINTENANCE)
# ==========================================

def gen_hr_maintenance():
    s = [("La candidature", "The application"), ("L'entretien", "The interview")]
    v = [("est validé(e)", "is validated"), ("est refusé(e)", "is rejected")]
    c = [("par les RH.", "by HR."), ("pour le poste.", "for the job.")]
    sub, vb, ctx = random.choice(s), random.choice(v), random.choice(c)
    return f"{sub[0]} {vb[0]} {ctx[0]}", f"{sub[1]} {vb[1]} {ctx[1]}", "HR_HIRING"

def gen_love_maintenance():
    phrases = [("Je t'aime.", "I love you."), ("Tu me manques.", "I miss you."), ("Bisous.", "Kisses.")]
    fr, en = random.choice(phrases)
    return fr, en, "LOVE"

def gen_gossip_maintenance():
    phrases = [("Il est nul.", "He is useless."), ("C'est un idiot.", "He is an idiot."), ("Elle m'énerve.", "She annoys me.")]
    fr, en = random.choice(phrases)
    return fr, en, "GOSSIP"

def gen_misc_maintenance():
    phrases = [("Ok ça marche.", "Ok works."), ("Dac.", "Ok."), ("Je vois.", "I see."), ("A plus.", "See ya.")]
    fr, en = random.choice(phrases)
    return fr, en, "MISC"

# ==========================================
# 🚀 EXÉCUTION DU SCÉNARIO DE GÉNÉRATION
# ==========================================

# Répartition ciblée
targets = {
    "BUSINESS": 1200,    # Priorité Critique
    "CYBER": 1000,       # Priorité Haute (DDoS)
    "INFRA": 1000,       # Priorité Haute (K8s/Linux)
    "TECH": 450,         # Maintenance
    "HR_HIRING": 450,    # Maintenance
    "LOVE": 300,         # Maintenance
    "GOSSIP": 300,       # Maintenance
    "MISC": 300          # Maintenance
}

generators = {
    "BUSINESS": gen_business_angry,
    "CYBER": gen_cyber_ddos,
    "INFRA": gen_infra_k8s,
    "TECH": gen_tech_maintenance,
    "HR_HIRING": gen_hr_maintenance,
    "LOVE": gen_love_maintenance,
    "GOSSIP": gen_gossip_maintenance,
    "MISC": gen_misc_maintenance
}

print("⏳ Génération chirurgicale en cours...")

for topic, count in targets.items():
    gen_func = generators[topic]
    # On génère un peu plus pour pallier aux doublons éventuels
    for _ in range(int(count * 1.2)):
        fr, en, t = gen_func()
        dataset.append({"Topic": t, "Phrase_FR": fr, "Phrase_EN": en})

df = pd.DataFrame(dataset)

# -----------------------------------------------------
# ✅ CORRECTION : FORÇAGE DE L'ORDRE DES COLONNES
# -----------------------------------------------------
df = df[['Phrase_FR', 'Phrase_EN', 'Topic']]

# Nettoyage
df = df.drop_duplicates(subset=['Phrase_FR'])
df = df.sample(frac=1).reset_index(drop=True)
df = df.head(TARGET_TOTAL)

print(f"✅ Dataset V10 (Ordered) généré : {len(df)} lignes.")
print("\nAperçu (colonnes vérifiées) :")
print(df.head(5))

# Export
df.to_csv(OUTPUT_FILE, index=False)

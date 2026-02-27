"""
Space Object Registration Compliance Pipeline (2020-2024)
=========================================================
Merges annual CSV files from UNOOSA Online Index and classifies
non-compliant (unregistered) space objects using RF, LGBM, and LR.

Usage:
    python space_compliance_pipeline.py

Requirements:
    pip install pandas numpy scikit-learn imbalanced-learn lightgbm shap matplotlib seaborn chardet

Input files (place in same folder as this script):
    2020.csv, 2021.csv, 2022.csv, 2023.csv, 2024.csv
    (2025.csv is excluded — only 1% registration rate due to filing lag)

Outputs (written to ./outputs/):
    plots/metrics_bar.png
    plots/confusion_matrices.png
    plots/roc_curves.png
    plots/shap_beeswarm.png
    plots/shap_bar.png
    exports/merged_features.csv
    exports/model_comparison.csv
    shap/shap_importance.csv
    models/*.pkl
"""

import os, sys, warnings, joblib
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import StandardScaler
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier
from sklearn.metrics            import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling     import SMOTE
import lightgbm as lgb
import shap

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, 'outputs')
TODAY       = pd.Timestamp('2026-02-27')   # snapshot date for LDINTERVAL

# File configs: year -> (encoding, rows to skip before header)
FILE_CONFIGS = {
    '2020': ('utf-8-sig', 1),   # blank row before header
    '2021': ('utf-8-sig', 0),
    '2022': ('ascii',     0),
    '2023': ('latin-1',   0),
    '2024': ('latin-1',   0),
    # 2025 intentionally excluded — filing lag contaminates the signal
}

# Leakage columns — filled in after/because of registration, not before
LEAKAGE_COLS = [
    'International Designator',   # unique ID, no predictive value
    'Name of Space Object',        # unique ID
    'Registration Document',       # exists only when registered
    'Function of Space Object',    # filled by UN after registration (99.97% correlated with target)
    "Secretariat`s Remarks",       # text literally says "Not registered..." or "submission processing..."
]

STATE_ALIAS = {
    'RUSSIAN FEDERATION': 'RUSSIA',
    'UNITED STATES':      'USA',
    'PEOPLES REPUBLIC OF CHINA': 'CHINA',
    'U.S.A':              'USA',
}

SEP = '=' * 62

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def hdr(title):
    print(f'\n{SEP}\n  {title}\n{SEP}')


def parse_date(s):
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip('[] ')
    for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y'):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            pass
    return pd.to_datetime(s, errors='coerce')


def make_dirs():
    for sub in ('plots', 'exports', 'models', 'shap'):
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)


def out(subdir, filename):
    return os.path.join(OUTPUT_DIR, subdir, filename)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — LOAD & MERGE
# ─────────────────────────────────────────────────────────────────────────────

def load_and_merge():
    hdr('PHASE 1 — Load & Merge Annual CSV Files')

    frames = []
    for yr, (enc, skip) in FILE_CONFIGS.items():
        path = os.path.join(SCRIPT_DIR, f'{yr}.csv')
        if not os.path.exists(path):
            print(f'  WARNING: {yr}.csv not found at {path} — skipping')
            continue

        # Read raw, no header, python engine tolerates ragged/malformed lines
        raw = pd.read_csv(path, dtype=str, encoding=enc,
                          engine='python', on_bad_lines='skip', header=None)

        # Skip rows before header
        if skip:
            raw = raw.iloc[skip:].reset_index(drop=True)

        # Use first row as column names
        raw.columns = [str(c).lstrip('\ufeff').strip() for c in raw.iloc[0]]
        raw = raw.iloc[1:].reset_index(drop=True)

        # Strip non-breaking spaces and whitespace from all string values
        for col in raw.columns:
            if raw[col].dtype == object:
                raw[col] = raw[col].str.replace('\xa0', '', regex=False).str.strip()

        raw['_year'] = yr

        # Drop blank separator rows (no registration status)
        raw = raw[raw['UN Registered'].notna()]

        yes = (raw['UN Registered'].str.lower() == 'yes').sum()
        no  = (raw['UN Registered'].str.lower() == 'no').sum()
        print(f'  {yr}.csv : {len(raw):>4} rows  |  Yes={yes:>4}  No={no:>3}  |  enc={enc}')
        frames.append(raw)

    if not frames:
        print('\nERROR: No CSV files loaded. Place 2020.csv–2024.csv in the same folder as this script.')
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    print(f'\n  Combined (before dedup) : {len(combined):,} rows')

    # Deduplicate on International Designator — keep first occurrence (earliest year)
    dedup = combined.drop_duplicates(subset=['International Designator'], keep='first')
    print(f'  After dedup             : {len(dedup):,} rows  '
          f'(removed {len(combined)-len(dedup):,} cross-year duplicates)')

    yes = (dedup['UN Registered'].str.lower() == 'yes').sum()
    no  = (dedup['UN Registered'].str.lower() == 'no').sum()
    print(f'\n  Final: Yes={yes:,}  No={no:,}  Imbalance={yes/no:.1f}:1  (minority = No = non-compliant)')

    print(f'\n  Year distribution:')
    for yr, cnt in dedup['_year'].value_counts().sort_index().items():
        print(f'    {yr}: {cnt:,} objects')

    return dedup


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — PREPROCESS & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df):
    hdr('PHASE 2 — Preprocess & Feature Engineering')

    # ── Leakage audit ────────────────────────────────────────────────────────
    print('\n  LEAKAGE AUDIT:')
    print('  ┌──────────────────────────────┬────────────────────────────────────┐')
    print('  │ Column                       │ Reason dropped                     │')
    print('  ├──────────────────────────────┼────────────────────────────────────┤')
    print('  │ Registration Document        │ Only exists when registered        │')
    print('  │ Function of Space Object     │ Filled AFTER registration (99.97%) │')
    print('  │ Secretariat Remarks          │ Text says "Not registered" directly │')
    print('  │ Intl. / Name of Space Object │ Unique IDs — no predictive value   │')
    print('  └──────────────────────────────┴────────────────────────────────────┘')

    for col in LEAKAGE_COLS + ['_year']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # ── Target ───────────────────────────────────────────────────────────────
    df = df.copy()
    df['REG'] = (df['UN Registered'].str.strip().str.lower() == 'yes').astype(int)
    df = df.drop(columns=['UN Registered'])
    rc = df['REG'].value_counts().sort_index()
    print(f'\n  Target:  Registered(1)={rc[1]:,}  Unregistered(0)={rc[0]:,}')
    print(f'  Minority class = Unregistered(0) = non-compliant objects')

    # ── Binary presence features ──────────────────────────────────────────────
    print(f'\n  Binary presence features (leakage-free — exist before registration):')
    print(f'  {"Feature":<8}  {"Source":<30}  {"=1":>6}  {"% present":>10}')
    print(f'  {"-"*60}')
    for feat, src in [('NDES',  'National Designator'),
                      ('GSO',   'GSO Location'),
                      ('ODOC',  'Other Documents'),
                      ('EXWEB', 'External website')]:
        df[feat] = df[src].notna().astype(int)
        df = df.drop(columns=[src])
        p = int(df[feat].sum())
        print(f'  {feat:<8}  {src:<30}  {p:>6,}  {p/len(df)*100:>9.1f}%')

    # ── DECAYED ──────────────────────────────────────────────────────────────
    df['DECAYED'] = df['Status'].str.lower().str.contains(
        'decay|reenter|re-enter', na=False).astype(int)
    df = df.drop(columns=['Status'])
    print(f'\n  DECAYED: {df["DECAYED"].sum():,} objects re-entered or decayed')

    # ── Temporal features ─────────────────────────────────────────────────────
    df['_ld']      = df['Date of Launch'].apply(parse_date)
    df['LDINTERVAL'] = (TODAY - df['_ld']).dt.days
    df['LDINTERVAL'].fillna(df['LDINTERVAL'].median(), inplace=True)
    df['L_YEAR']   = df['_ld'].dt.year.fillna(2022).astype(int)
    df = df.drop(columns=['Date of Launch', 'Date of Decay or Change',
                          '_ld'], errors='ignore')
    print(f'\n  LDINTERVAL (days from launch to {TODAY.date()}):')
    print(f'    min={df["LDINTERVAL"].min():.0f}d  '
          f'median={df["LDINTERVAL"].median():.0f}d  '
          f'max={df["LDINTERVAL"].max():.0f}d')
    print(f'  L_YEAR: {sorted(df["L_YEAR"].unique())}')

    # ── State encoding ────────────────────────────────────────────────────────
    df['_st'] = (
        df['State/Organization']
        .str.strip().str.strip('[]() ').str.upper()
        .str.replace(r'\s+', ' ', regex=True)
        .fillna('UNKNOWN').replace(STATE_ALIAS)
    )
    df = df.drop(columns=['State/Organization'])

    top5 = df['_st'].value_counts().nlargest(5).index.tolist()
    print(f'\n  Non-compliance rate by state (top 10 by volume):')
    print(f'  {"State":<28}  {"Total":>6}  {"NC(No)":>7}  {"NC%":>6}  {"TOP5"}')
    print(f'  {"-"*60}')
    stats = df.groupby('_st').agg(
        total=('REG','count'), nc=('REG', lambda x: (x==0).sum())
    ).sort_values('total', ascending=False)
    for st, row in stats.head(10).iterrows():
        flag = ' ✓' if st in top5 else ''
        print(f'  {st:<28}  {row.total:>6,}  {row.nc:>7,}  '
              f'{row.nc/row.total*100:>5.1f}%{flag}')

    df['_st_enc'] = df['_st'].apply(lambda x: x if x in top5 else 'OTHER')
    dummies = pd.get_dummies(df['_st_enc'], prefix='STATE', drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=['_st', '_st_enc'])

    # ── Final cleanup ─────────────────────────────────────────────────────────
    feat_cols = [c for c in df.columns if c != 'REG']
    df[feat_cols] = df[feat_cols].fillna(0)

    print(f'\n  Final feature set ({len(feat_cols)}):')
    for f in feat_cols:
        nz   = (df[f] != 0).sum()
        kind = 'continuous' if f in ('LDINTERVAL', 'L_YEAR') else 'binary'
        print(f'    {f:<30}  non-zero={nz:>5,}  {kind}')

    assert not df[feat_cols].isnull().any().any(), 'NaN values remain after fillna!'
    df.to_csv(out('exports', 'merged_features.csv'), index=False)
    print(f'\n  Saved -> outputs/exports/merged_features.csv')

    return df, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — SPLIT & IMBALANCE STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

def prepare_splits(df, feat_cols):
    hdr('PHASE 3 — Train/Test Split & Imbalance Strategies')

    X = df[feat_cols]
    y = df['REG']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    print(f'\n  Stratified 80/20 split:')
    print(f'  {"Set":<8}  {"Rows":>6}  {"Reg(1)":>8}  {"Unreg(0)":>9}  {"Ratio(1:0)":>12}')
    print(f'  {"-"*52}')
    for name, ys in [('Train', y_tr), ('Test', y_te)]:
        c1 = int(ys.sum()); c0 = int((ys==0).sum())
        print(f'  {name:<8}  {len(ys):>6,}  {c1:>8,}  {c0:>9,}  {c1/max(c0,1):>10.1f}:1')

    scaler   = StandardScaler()
    X_tr_sc  = pd.DataFrame(scaler.fit_transform(X_tr), columns=feat_cols)
    X_te_sc  = pd.DataFrame(scaler.transform(X_te),     columns=feat_cols)

    # SMOTE: k_neighbors=3 because minority class is small (~142 train samples)
    smote    = SMOTE(random_state=42, k_neighbors=3)
    X_sm,    y_sm    = smote.fit_resample(X_tr,    y_tr)
    X_sm_sc, y_sm_sc = smote.fit_resample(X_tr_sc, y_tr)

    cw      = compute_class_weight('balanced', classes=np.array([0,1]), y=y_tr)
    cw_dict = {0: float(cw[0]), 1: float(cw[1])}

    print(f'\n  Imbalance strategies:')
    sm_nc = int((y_sm==0).sum())
    print(f'    1. Baseline    : {len(y_tr):,} rows, original distribution')
    print(f'    2. SMOTE       : {len(y_sm):,} rows  (Unreg={sm_nc:,}  Reg={len(y_sm)-sm_nc:,})')
    print(f'    3. ClassWeight : w[Unreg(0)]={cw[0]:.2f}  w[Reg(1)]={cw[1]:.2f}')
    print(f'       (Unregistered/minority is upweighted {cw[0]/cw[1]:.0f}x)')

    # Class distribution plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (title, ys) in zip(axes, [
        ('Baseline\n(train)',  y_tr),
        ('SMOTE\n(train)',     y_sm),
        ('Test set',           y_te),
    ]):
        cnts = pd.Series(ys).value_counts().sort_index()
        bars = ax.bar(['Reg(1)', 'Unreg(0)'], [cnts.get(1,0), cnts.get(0,0)],
                      color=['#3498db','#e74c3c'], edgecolor='black', width=0.5)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                    f'{int(bar.get_height()):,}', ha='center', fontsize=9, fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Count'); ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Class Distribution Across Imbalance Strategies', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out('plots', 'class_distributions.png'), dpi=150); plt.close()
    print(f'\n  Plot -> outputs/plots/class_distributions.png')

    return (X_tr, X_te, y_tr, y_te,
            X_tr_sc, X_te_sc,
            X_sm, y_sm, X_sm_sc, y_sm_sc,
            cw_dict, scaler)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — TRAIN & EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(name, clf, X_train, y_train, X_test, y_test, feat_cols):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test,  y_pred)
    # Metrics reported for minority/focus class: Unregistered(0)
    prec = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec  = recall_score(y_test,    y_pred, pos_label=0, zero_division=0)
    f1   = f1_score(y_test,        y_pred, pos_label=0, zero_division=0)
    auc  = roc_auc_score(y_test,   y_prob)
    cm   = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    rep = classification_report(
        y_test, y_pred,
        target_names=['Unregistered(0)', 'Registered(1)'],
        output_dict=True, zero_division=0
    )

    print(f'\n  {"="*58}')
    print(f'  MODEL: {name}')
    print(f'  {"="*58}')
    print(f'  Accuracy : {acc:.4f}  |  ROC-AUC: {auc:.4f}')
    print(f'\n  Per-class metrics:')
    print(f'  {"Class":<20} {"Precision":>10} {"Recall":>8} {"F1":>8} {"Support":>9}')
    print(f'  {"-"*58}')
    for lbl in ['Unregistered(0)', 'Registered(1)']:
        r = rep[lbl]
        note = '  <-- FOCUS (non-compliant)' if lbl == 'Unregistered(0)' else ''
        print(f'  {lbl:<20} {r["precision"]:>10.4f} {r["recall"]:>8.4f} '
              f'{r["f1-score"]:>8.4f} {r["support"]:>9.0f}{note}')
    print(f'  {"Macro avg":<20} '
          f'{rep["macro avg"]["precision"]:>10.4f} '
          f'{rep["macro avg"]["recall"]:>8.4f} '
          f'{rep["macro avg"]["f1-score"]:>8.4f}')

    print(f'\n  Confusion Matrix:')
    print(f'  {"":>24}  Pred:Unreg  Pred:Reg')
    print(f'  {"Actual:Unreg(NC)":>24}  {tn:>10,}  {fp:>8,}  '
          f'(TN=correctly flagged | FP=NC missed)')
    print(f'  {"Actual:Reg":>24}  {fn:>10,}  {tp:>8,}')

    joblib.dump(clf, out('models', f'{name}.pkl'))

    return {
        'Model': name, 'Accuracy': acc,
        'Prec_NC': prec, 'Rec_NC': rec, 'F1_NC': f1, 'ROC_AUC': auc,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        '_clf': clf, '_y_prob': y_prob
    }


def run_experiments(splits, feat_cols, y_te):
    hdr('PHASE 4 — Train & Evaluate (9 Experiments)')
    print('  Focus metric: F1 and Recall for Unregistered(0) = non-compliant\n')

    (X_tr, X_te, y_tr, y_te_,
     X_tr_sc, X_te_sc,
     X_sm, y_sm, X_sm_sc, y_sm_sc,
     cw_dict, scaler) = splits

    lgbm_spw = cw_dict[0] / cw_dict[1]   # upweight minority class 0

    experiments = [
        ('LR_Baseline',
         LogisticRegression(C=1, max_iter=1000, solver='lbfgs', random_state=42),
         X_tr_sc, y_tr),
        ('LR_SMOTE',
         LogisticRegression(C=1, max_iter=1000, solver='lbfgs', random_state=42),
         X_sm_sc, y_sm_sc),
        ('LR_ClassWeight',
         LogisticRegression(C=1, max_iter=1000, solver='lbfgs',
                            class_weight='balanced', random_state=42),
         X_tr_sc, y_tr),
        ('RF_Baseline',
         RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
         X_tr, y_tr),
        ('RF_SMOTE',
         RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
         X_sm, y_sm),
        ('RF_ClassWeight',
         RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                random_state=42, n_jobs=-1),
         X_tr, y_tr),
        ('LGBM_Baseline',
         lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                            verbosity=-1, random_state=42),
         X_tr, y_tr),
        ('LGBM_SMOTE',
         lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                            verbosity=-1, random_state=42),
         X_sm, y_sm),
        ('LGBM_ClassWeight',
         lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                            scale_pos_weight=lgbm_spw,
                            verbosity=-1, random_state=42),
         X_tr, y_tr),
    ]

    all_results = []
    for name, clf, Xtr, ytr in experiments:
        res = evaluate_model(name, clf, Xtr, ytr, X_te, y_te, feat_cols)
        all_results.append(res)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — COMPARATIVE SUMMARY & PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def summarise_and_plot(all_results, y_te):
    hdr('PHASE 5 — Comparative Summary')

    cols  = ['Model','Accuracy','Prec_NC','Rec_NC','F1_NC','ROC_AUC','TN','FP','FN','TP']
    sumdf = (pd.DataFrame(all_results)[cols]
             .sort_values('F1_NC', ascending=False)
             .reset_index(drop=True))
    sumdf.index += 1

    print(f'\n  Ranked by F1 (Unregistered/non-compliant class):\n')
    print(f'  {"Rk":<4} {"Model":<22} {"Acc":>6} {"Prec_NC":>8} {"Rec_NC":>7} '
          f'{"F1_NC":>6} {"AUC":>6}  {"TN":>3} {"FP":>3}')
    print(f'  {"-"*75}')
    for rk, row in sumdf.iterrows():
        print(f'  {rk:<4} {row.Model:<22} {row.Accuracy:>6.4f} {row.Prec_NC:>8.4f} '
              f'{row.Rec_NC:>7.4f} {row.F1_NC:>6.4f} {row.ROC_AUC:>6.4f}  '
              f'{row.TN:>3.0f} {row.FP:>3.0f}')
    print(f'\n  TN = non-compliant correctly flagged  |  FP = non-compliant missed')

    print(f'\n  Strategy comparison (mean across 3 algorithms):')
    for strat in ['Baseline', 'SMOTE', 'ClassWeight']:
        sub = sumdf[sumdf.Model.str.contains(strat)]
        print(f'    {strat:<13}: F1={sub.F1_NC.mean():.4f}  '
              f'Rec={sub.Rec_NC.mean():.4f}  Prec={sub.Prec_NC.mean():.4f}  '
              f'TN={sub.TN.mean():.0f}  FP={sub.FP.mean():.0f}')

    print(f'\n  Algorithm comparison (mean across 3 strategies):')
    for algo in ['LR', 'RF', 'LGBM']:
        sub = sumdf[sumdf.Model.str.startswith(algo)]
        print(f'    {algo:<6}: F1={sub.F1_NC.mean():.4f}  AUC={sub.ROC_AUC.mean():.4f}')

    best = sumdf.iloc[0]
    print(f'\n  BEST MODEL: {best.Model}')
    print(f'    Prec_NC : {best.Prec_NC:.4f}')
    print(f'    Rec_NC  : {best.Rec_NC:.4f}')
    print(f'    F1_NC   : {best.F1_NC:.4f}')
    print(f'    ROC_AUC : {best.ROC_AUC:.4f}')
    print(f'    TN      : {best.TN:.0f}  (NC objects correctly flagged)')
    print(f'    FP      : {best.FP:.0f}  (NC objects missed)')

    sumdf.to_csv(out('exports', 'model_comparison.csv'), index=False)
    print(f'\n  Saved -> outputs/exports/model_comparison.csv')

    # Metrics bar chart
    metrics = ['Prec_NC', 'Rec_NC', 'F1_NC', 'ROC_AUC']
    colors  = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    x, w    = np.arange(len(sumdf)), 0.20
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (m, c) in enumerate(zip(metrics, colors)):
        ax.bar(x + i*w, sumdf[m], w, label=m, color=c, edgecolor='black', alpha=0.85)
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels(sumdf.Model, rotation=30, ha='right', fontsize=8)
    ax.axhline(0.80, color='orange', linestyle='--', alpha=0.6, label='0.80 reference')
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score')
    ax.set_title('2020–2024 Multi-Year | Non-Compliance Detection — 9 Experiments',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out('plots', 'metrics_bar.png'), dpi=150); plt.close()
    print('  Plot -> outputs/plots/metrics_bar.png')

    # Confusion matrices
    n, ncols = len(all_results), 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
    flat = axes.flatten()
    for i, res in enumerate(all_results):
        cm = np.array([[res['TN'], res['FP']], [res['FN'], res['TP']]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=flat[i],
                    xticklabels=['Pred:Unreg', 'Pred:Reg'],
                    yticklabels=['Act:Unreg', 'Act:Reg'], linewidths=0.5)
        flat[i].set_title(
            f"{res['Model']}\nF1_NC={res['F1_NC']:.3f}  AUC={res['ROC_AUC']:.3f}",
            fontsize=9, fontweight='bold')
    for ax in flat[n:]:
        ax.set_visible(False)
    plt.suptitle('Confusion Matrices — Non-Compliance Detection (2020–2024)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out('plots', 'confusion_matrices.png'), dpi=150); plt.close()
    print('  Plot -> outputs/plots/confusion_matrices.png')

    # ROC curves
    algo_colors = {'LR': '#2196F3', 'RF': '#4CAF50', 'LGBM': '#FF9800'}
    fig, ax = plt.subplots(figsize=(9, 7))
    for res in all_results:
        algo = res['Model'].split('_')[0]
        fpr, tpr, _ = roc_curve(y_te, res['_y_prob'])
        ax.plot(fpr, tpr,
                label=f"{res['Model']} ({res['ROC_AUC']:.3f})",
                color=algo_colors.get(algo, 'gray'), alpha=0.75)
    ax.plot([0,1],[0,1], 'k--', alpha=0.4, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — 2020–2024 Dataset', fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out('plots', 'roc_curves.png'), dpi=150); plt.close()
    print('  Plot -> outputs/plots/roc_curves.png')

    return sumdf, best.Model


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────

def explain_best(best_name, all_results, X_te, feat_cols):
    hdr(f'PHASE 6 — SHAP Explainability ({best_name})')

    best_res = next(r for r in all_results if r['Model'] == best_name)
    clf      = best_res['_clf']
    algo     = best_name.split('_')[0]

    print(f'  Algorithm : {algo}')

    if algo in ('RF', 'LGBM'):
        explainer = shap.TreeExplainer(clf)
        sv        = explainer.shap_values(X_te)
        if isinstance(sv, list):
            sv = sv[1]
        elif hasattr(sv, 'shape') and len(sv.shape) == 3:
            sv = sv[:, :, 1]
    else:
        explainer = shap.LinearExplainer(clf, X_te)
        sv        = explainer.shap_values(X_te)

    mean_abs = np.abs(sv).mean(axis=0)
    imp = (pd.DataFrame({'Feature': feat_cols, 'Mean|SHAP|': mean_abs})
           .sort_values('Mean|SHAP|', ascending=False)
           .reset_index(drop=True))

    print(f'\n  Global feature importance (mean |SHAP value|):\n')
    print(f'  {"Rk":<4} {"Feature":<30} {"Mean|SHAP|":>12}  Bar')
    print(f'  {"-"*62}')
    mv = imp['Mean|SHAP|'].max()
    for rk, row in imp.iterrows():
        bar = chr(9608) * int(row['Mean|SHAP|'] / max(mv, 1e-9) * 30)
        print(f'  {rk+1:<4} {row.Feature:<30} {row["Mean|SHAP|"]:>12.4f}  {bar}')

    imp.to_csv(out('shap', 'shap_importance.csv'), index=False)

    # Beeswarm
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_te.values, feature_names=feat_cols,
                      plot_type='dot', show=False)
    plt.title(f'SHAP Beeswarm — {best_name} (2020–2024)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(out('plots', 'shap_beeswarm.png'), dpi=150); plt.close()

    # Bar
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_te.values, feature_names=feat_cols,
                      plot_type='bar', show=False)
    plt.title(f'SHAP Feature Importance — {best_name}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(out('plots', 'shap_bar.png'), dpi=150); plt.close()

    print(f'\n  Saved -> outputs/shap/shap_importance.csv')
    print(f'  Saved -> outputs/plots/shap_beeswarm.png')
    print(f'  Saved -> outputs/plots/shap_bar.png')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f'\n{SEP}')
    print(f'  SPACE OBJECT REGISTRATION COMPLIANCE PIPELINE')
    print(f'  UNOOSA 2020–2024  |  Sawmiller (2024) alignment')
    print(f'{SEP}')

    make_dirs()

    df_raw              = load_and_merge()
    df_feat, feat_cols  = preprocess(df_raw)
    splits              = prepare_splits(df_feat, feat_cols)
    y_te                = splits[3]
    all_results         = run_experiments(splits, feat_cols, y_te)
    sumdf, best_name    = summarise_and_plot(all_results, y_te)
    X_te                = splits[1]
    explain_best(best_name, all_results, X_te, feat_cols)

    print(f'\n{SEP}')
    print(f'  PIPELINE COMPLETE')
    print(f'  Best model : {best_name}')
    print(f'  All outputs: {OUTPUT_DIR}/')
    print(f'{SEP}\n')


if __name__ == '__main__':
    main()
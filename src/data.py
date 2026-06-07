"""Wczytywanie i preprocessing zbiorow UNSW-NB15 oraz CIC-IoT2023."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

DATA_DIR = "data"
SEED = 42

NSLKDD_COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack", "difficulty",
]

NSLKDD_CATEGORY = {
    "normal": "Normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS", "processtable": "DoS",
    "worm": "DoS", "mailbomb": "DoS",
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R", "perl": "U2R",
    "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}


@dataclass
class Dataset:
    name: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    features: list
    classes: list


def _encode_categorical(train, test):
    cat = [c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c])]
    for c in cat:
        le = LabelEncoder().fit(pd.concat([train[c], test[c]]).astype(str))
        train[c] = le.transform(train[c].astype(str))
        test[c] = le.transform(test[c].astype(str))
    return train, test


def _drop_constant(train, test):
    const = [c for c in train.columns if train[c].nunique() <= 1]
    return train.drop(columns=const), test.drop(columns=const)


def _finalize(train, test, target, name, max_train):
    train = train.replace([np.inf, -np.inf], np.nan).dropna()
    test = test.replace([np.inf, -np.inf], np.nan).dropna()

    if max_train and len(train) > max_train:
        train = train.groupby(target, group_keys=False).sample(
            frac=max_train / len(train), random_state=SEED)

    y_enc = LabelEncoder().fit(pd.concat([train[target], test[target]]).astype(str))
    y_train = y_enc.transform(train[target].astype(str))
    y_test = y_enc.transform(test[target].astype(str))

    X_train = train.drop(columns=[target])
    X_test = test.drop(columns=[target])
    X_train, X_test = _encode_categorical(X_train, X_test)
    X_train, X_test = _drop_constant(X_train, X_test)

    scaler = MinMaxScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return Dataset(name, X_train, X_test, y_train, y_test,
                   list(X_train.columns), list(y_enc.classes_))


def load_unsw(max_train=None):
    train = pd.read_csv(f"{DATA_DIR}/unsw_train.csv").drop(columns=["id", "label"])
    test = pd.read_csv(f"{DATA_DIR}/unsw_test.csv").drop(columns=["id", "label"])
    return _finalize(train, test, "attack_cat", "UNSW-NB15", max_train)


def load_ciciot(max_train=150000):
    drop = ["Label", "label"]
    train = pd.read_parquet(f"{DATA_DIR}/ciciot_train.parquet").drop(columns=drop)
    test = pd.read_parquet(f"{DATA_DIR}/ciciot_test.parquet").drop(columns=drop)
    return _finalize(train, test, "attack_class", "CIC-IoT2023", max_train)


def load_nslkdd(max_train=None):
    parts = [pd.read_csv(f"{DATA_DIR}/{f}.txt", header=None, names=NSLKDD_COLS)
             for f in ("KDDTrain+", "KDDTest+")]
    df = pd.concat(parts, ignore_index=True).drop(columns=["difficulty"])
    df["attack"] = df["attack"].map(NSLKDD_CATEGORY)
    train, test = train_test_split(
        df, test_size=0.3, random_state=SEED, stratify=df["attack"])
    return _finalize(train, test, "attack", "NSL-KDD", max_train)


LOADERS = {"UNSW-NB15": load_unsw, "CIC-IoT2023": load_ciciot}
BASELINE_LOADERS = {"NSL-KDD": load_nslkdd}

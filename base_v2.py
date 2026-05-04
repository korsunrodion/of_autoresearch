import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from db_v2 import db, TrackingLinkSubscriber

# Risk levels are lowercase in tracking_links_subscriber
RISK_MAP = {
    'no risk':   1,
    'low':       2,
    'high':      3,
    'very high': 4,
    'extreme':   5,
}

# Reverse map: internal title-case → DB lowercase
RISK_TO_DB = {
    'No risk':   'no risk',
    'Low':       'low',
    'High':      'high',
    'Very High': 'very high',
    'Extreme':   'extreme',
}


def fetch_df(
    tracking_link_id: str | list[str] | None = None,
    processed: bool | None = None,
) -> pd.DataFrame:
    db.connect(reuse_if_open=True)
    query = TrackingLinkSubscriber.select()

    if processed is not None:
        # Table has no is_processed column yet; guard against AttributeError
        if hasattr(TrackingLinkSubscriber, 'is_processed'):
            query = query.where(TrackingLinkSubscriber.is_processed == processed)

    if isinstance(tracking_link_id, list):
        query = query.where(TrackingLinkSubscriber.tracking_link_id.in_(tracking_link_id))
    elif tracking_link_id:
        query = query.where(TrackingLinkSubscriber.tracking_link_id == tracking_link_id)

    return pd.DataFrame(list(query.dicts()))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Rename to internal names used throughout all classifier scripts
    df = df.rename(columns={
        'username':          'user_name',
        'tracking_link_id':  'tracking_model_name',
        'subscription_date': 'subscribed_at',
        'user_id':           'user_id_num',
    })

    df['subscribed_at'] = pd.to_datetime(df['subscribed_at'], errors='coerce')
    df = df.dropna(subset=['subscribed_at', 'user_name'])

    # Normalise risk_level to title-case so classifiers work unchanged
    df['risk_level'] = df['risk_level'].str.title().replace({'No Risk': 'No risk'})
    df['risk_score'] = df['risk_level'].map({
        'No risk': 1, 'Low': 2, 'High': 3, 'Very High': 4, 'Extreme': 5,
    })
    df = df.dropna(subset=['risk_score'])

    df['subscribed_ts']     = df['subscribed_at'].astype('int64') // 10 ** 9
    if 'total_chargebacks' in df.columns:
        df['total_chargebacks'] = pd.to_numeric(df['total_chargebacks'], errors='coerce').fillna(0)
    else:
        df['total_chargebacks'] = 0

    cols = ['user_name', 'tracking_model_name', 'subscribed_at',
            'user_id_num', 'subscribed_ts', 'risk_level', 'risk_score',
            'total_chargebacks']
    if 'is_internal_data' in df.columns:
        cols.append('is_internal_data')
    return df[cols].dropna(subset=['user_id_num'])


def update_risk_levels(predictions: dict[str, str]) -> int:
    """
    Write predicted risk levels back to the DB.

    Args:
        predictions: {username: predicted_risk} where predicted_risk is
                     title-case ('No risk', 'Low', 'High', 'Very High', 'Extreme').

    Returns:
        Number of rows updated.
    """
    db.connect(reuse_if_open=True)

    updated = 0
    # Batch into chunks of 500 to avoid overly large queries
    items = list(predictions.items())
    chunk_size = 500
    for i in range(0, len(items), chunk_size):
        chunk = items[i:i + chunk_size]
        for username, risk_title in chunk:
            risk_db = RISK_TO_DB.get(risk_title)
            if risk_db is None:
                continue
            n = (
                TrackingLinkSubscriber
                .update(risk_level=risk_db)
                .where(
                    (TrackingLinkSubscriber.username == username) &
                    (TrackingLinkSubscriber.is_internal_data != True)
                )
                .execute()
            )
            updated += n

    return updated

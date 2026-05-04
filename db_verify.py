from datetime import datetime
import os
from dotenv import load_dotenv
from peewee import PostgresqlDatabase, Model, AutoField, CharField, BooleanField, DateTimeField
from urllib.parse import urlparse

load_dotenv()


def _make_verify_db():
    url = os.environ.get('DB_VERIFY_URL')
    if not url:
        raise RuntimeError('DB_VERIFY_URL environment variable not set')
    p = urlparse(url)
    return PostgresqlDatabase(
        p.path.lstrip('/'),
        user=p.username,
        password=p.password or '',
        host=p.hostname,
        port=p.port or 5432,
    )


verify_db = _make_verify_db()


class Subscription(Model):
    id                  = AutoField()
    user_id             = CharField(null=True)
    user_name           = CharField(null=True)
    subscribed_at       = CharField(null=True)
    timeline            = CharField(null=True)
    total_chargebacks   = CharField(null=True)
    total               = CharField(null=True)
    interrupted         = CharField(null=True)
    risk_level          = CharField(null=True)
    v1                  = BooleanField(null=True)
    selected            = BooleanField(null=True)
    model_id            = CharField(null=True)
    tracking_model_id   = CharField(null=True)
    tracking_model_name = CharField(null=True)
    updated_at          = DateTimeField(default=datetime.now)

    class Meta:
        database   = verify_db
        table_name = 'subscriptions'

import os
from dotenv import load_dotenv
from peewee import PostgresqlDatabase, Model, CharField, IntegerField, BooleanField
from urllib.parse import urlparse

load_dotenv()


def _make_db():
    url = os.environ.get('DATABASE_URL')
    if url:
        p = urlparse(url)
        return PostgresqlDatabase(
            p.path.lstrip('/'),
            user=p.username,
            password=p.password or '',
            host=p.hostname,
            port=p.port or 5432,
        )
    return PostgresqlDatabase(
        os.environ.get('DB_NAME', 'of_parser'),
        user=os.environ.get('DB_USER', 'postgres'),
        password=os.environ.get('DB_PASSWORD', ''),
        host=os.environ.get('DB_HOST', 'localhost'),
        port=int(os.environ.get('DB_PORT', 5432)),
    )


db = _make_db()


class TrackingLinkSubscriber(Model):
    # Peewee attribute names match actual DB column names (snake_case, TypeORM default)
    id               = CharField(primary_key=True)
    tracking_link_id = CharField(null=True)
    username         = CharField(null=True)
    user_id          = IntegerField(null=True)
    subscription_date = CharField(null=True)
    risk_level       = CharField(null=True)
    is_processed      = BooleanField(default=False)
    total_chargebacks = IntegerField(default=0)
    is_internal_data  = BooleanField(null=True, default=False)

    class Meta:
        database   = db
        table_name = 'tracking_links_subscriber'

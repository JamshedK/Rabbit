#!/bin/bash
set -e

echo "📦 Removing corrupted postgresql.auto.conf..."
sudo rm -f /var/lib/postgresql/14/main/postgresql.auto.conf


echo "🔁 Restarting PostgreSQL 14 ..."
sudo pg_ctlcluster 14 main restart

echo "RAN command pg_ctlcluster 14 main restart"

echo "Waiting 5 seconds for initial startup..."
sleep 5

echo "⏳ Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if pg_isready -p 5432 -q; then
        echo "✅ PostgreSQL 14 is ready!"
        exit 0
    fi
    echo "⏳ Waiting... ($i/30)"
    sleep 2
done

echo "❌ PostgreSQL 14 failed to start properly"
exit 1
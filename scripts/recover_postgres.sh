#!/bin/bash
set -e

echo "📦 Removing corrupted postgresql.auto.conf..."
sudo rm -f /var/lib/postgresql/14/main/postgresql.auto.conf

echo "⏳ Waiting briefly..."
sleep 2

echo "🔁 Restarting PostgreSQL using pg_ctlcluster..."
sudo pg_ctlcluster 14 main restart

echo "✅ PostgreSQL restarted successfully."

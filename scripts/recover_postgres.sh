sudo rm /home/ubuntu/pgsql/bin/RAGTuner/postgresql.auto.conf
sleep 2
su - ubuntu -c '/home/ubuntu/pgsql/bin/pg_ctl restart -D /home/ubuntu/pgsql/bin/RAGTuner -o "-c config_file=/home/ubuntu/pgsql/bin/RAGTuner/postgresql.conf"'
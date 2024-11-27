ps aux | grep mysqld | grep my.cnf | awk '{print $2}'|xargs kill -9
sleep 5
echo "ycc" | sudo -S service mysql start


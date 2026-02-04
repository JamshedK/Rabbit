mkdir ../optimization_results/
mkdir ../optimization_results/temp_results/
mkdir ../optimization_results/$1/
mkdir ../optimization_results/$1/log/
sudo apt-get update
sudo apt-get install git
git clone https://github.com/cmu-db/benchbase.git ../benchbase
cd ../benchbase
git checkout b4b36683afdf79dd8bf70b95199b874c68218975
./mvnw clean package -P $1
cd target
tar xvzf benchbase-$1.tgz
cd benchbase-$1

START_TIME=$(date)
START_UNIX=$(date +%s)
echo "Start: $START_TIME (Unix: $START_UNIX)"

cd ../knowledge
cp ../.env ./postgres/graph/.env
graphrag index --root ./postgres/graph

END_TIME=$(date)
END_UNIX=$(date +%s)
DURATION=$((END_UNIX - START_UNIX))

echo "End: $END_TIME (Unix: $END_UNIX)"
echo "Duration: ${DURATION}s"
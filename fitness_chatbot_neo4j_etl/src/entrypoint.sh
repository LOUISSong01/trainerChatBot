#!/bin/bash

# Run any setup steps or pre-processing tasks here
echo "Running ETL to move data from CSV files to Neo4j"

#Run the ETL script 
python3 trainer_bulk_csv_write.py

echo "ETL process completed successfully"
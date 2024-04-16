#/bin/bash

cd ..
echo "Getting principal components of the images"
python transformation_flow.py -i "data/nachito*" --batch
python transformation_flow.py -i "data/marina*" --batch

echo Created files \[ $(ls components | grep .csv) \]
echo "Finding neighbors..."
python calculate_PCA_neigh.py "components/marina*.csv" "components/nachito*.csv"

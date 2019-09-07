export PROJECT_DIR="/home/amillert/private/readmesomebook/"
export NNRUN_PATH="${PROJECT_DIR}nnrunner.py "
export IN_PATH="${PROJECT_DIR}data/books/the-governors-man.txt"
export SW_PATH="${PROJECT_DIR}data/unique-stop-words/english-unique-sw.txt"

python $NNRUN_PATH -in $IN_PATH -sw $SW_PATH -w 4 -b 128 -e 1e-2 --epochs 100

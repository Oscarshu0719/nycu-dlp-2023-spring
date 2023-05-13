FILEID="1v86ALdPuxilagCl05PPqi6X_X9Ke3cwb"
FILENAME="processed_data.zip"
FILE_TMP="/tmp/cookies.txt"
DIR_UNZIP="./datasets/"

wget --load-cookies $FILE_TMP \
    "https://docs.google.com/uc?export=download&confirm= \
    $(wget --quiet --save-cookies ${FILE_TMP} --keep-session-cookies --no-check-certificate \
    "https://docs.google.com/uc?export=download&id=${FILEID}" -O- | \
    sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && \ 
unzip $FILENAME -d $DIR_UNZIP && \
rm $FILE_TMP $FILENAME

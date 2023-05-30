FILE_ID="1vAo33mix4H6h-00zkrNDRfCpusxlTWxU"
FILE_NAME="iclevr.zip"
FILE_TMP="/tmp/cookies.txt"
DIR_UNZIP="./datasets/"

wget --load-cookies $FILE_TMP \
    "https://docs.google.com/uc?export=download&confirm= \
    $(wget --quiet --save-cookies ${FILE_TMP} --keep-session-cookies --no-check-certificate \
    "https://docs.google.com/uc?export=download&id=${FILE_ID}" -O- | \
    sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILE_ID}" -O $FILE_NAME && \ 
unzip $FILE_NAME -d $DIR_UNZIP && \
rm $FILE_NAME

for i in *.MOV
do 
    ffmpeg -i "$i" -c:a aac -b:a 128k -c:v libx264 -crf 20 "${i%.MOV}.avi"
done
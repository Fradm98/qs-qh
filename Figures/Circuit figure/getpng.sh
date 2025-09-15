# Uso: ./getpng.sh source_file.tex target_file.extension

cwd=$PWD
outname=$(echo "$cwd/$1" | cut -f 1 -d '.')
pdflatex "$cwd/$1"
convert -density 600 "$outname.pdf" -quality 200 "$2"
#convert "$2" -channel RGB -negate "$2"
rm "$outname.aux"
rm "$outname.log"
rm "$outname.pdf"

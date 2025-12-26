#!/bin/bash

DURATION=2000
BASENAME="video_"
COUNT_FILE="count.txt"

# Initialize counter
if [ ! -f "$COUNT_FILE" ]; then
    echo 1 > "$COUNT_FILE"
fi

echo "--------------------------------------"
echo "Press ENTER to record a new video"
echo "Press BACKSPACE to exit"
echo "--------------------------------------"

while true; do
    # Read a single key silently
    read -rsn1 KEY

    # ENTER key (newline)
    if [[ "$KEY" == "" ]]; then
        COUNT=$(cat "$COUNT_FILE")
        OUTPUT="${BASENAME}${COUNT}.mp4"

        echo
        echo "Recording $OUTPUT ..."

        rpicam-vid -t $DURATION -o "$OUTPUT"

        echo "Recording finished"
        echo $((COUNT + 1)) > "$COUNT_FILE"

        echo "Press ENTER to record again, BACKSPACE to exit"

    # BACKSPACE key (ASCII 127)
    elif [[ "$KEY" == $'\x7f' ]]; then
        echo
        echo "Exiting script."
        exit 0
    fi
done

#!/bin/bash

# Directory containing clean images
input_dir="../data/train/train/data"
# Directory to save stego images
output_dir="../data/train/train/stego"
# List of messages to embed
messages=("Hello Everyone!" "This is a secret message." "Steganography is fun!" "Data hiding example." "Keep it safe!" "Hidden in plain sight." "Confidential information." "Secure your data." "This is a test." "Embedding data in images."
    "The quick brown fox jumps over the lazy dog." "The sun is shining brightly today." "I love to read books in my free time." "The beach is my favorite vacation spot." "I am excited for the weekend." "The city is always bustling with activity." "The mountains are a great place to hike." "I love to try new foods." "The park is a great place to relax." "I am looking forward to the future."
    "The world is a beautiful place." "I love to learn new things." "The ocean is a vast and mysterious place." "I am grateful for my friends and family." "The forest is a great place to explore." "I love to listen to music." "The desert is a harsh and unforgiving environment." "I am excited for the new year." "The city is a great place to live." "I love to travel to new places."
    "The snow is falling gently outside." "I love to play sports in my free time." "The river is a great place to fish." "I am looking forward to the holidays." "The lake is a beautiful place to relax." "I love to read poetry." "The wind is blowing strongly today." "I am grateful for the simple things in life." "The earth is a fragile and beautiful place." "I love to learn about history."
    "The stars are shining brightly in the sky." "I love to play with my pets." "The rain is falling gently outside." "I am excited for the summer." "The flowers are blooming beautifully." "I love to listen to podcasts." "The thunder is rumbling loudly." "I am looking forward to the weekend." "The sun is setting slowly in the sky." "I love to try new restaurants."
    "The world is a complex and interesting place." "I love to learn about science." "The ocean is a powerful and mysterious place." "I am grateful for my health." "The forest is a great place to go on an adventure." "I love to read fiction." "The desert is a harsh and beautiful environment." "I am excited for the future." "The city is a great place to meet new people." "I love to travel to new countries."
    "The snow is falling heavily outside." "I love to play board games in my free time." "The river is a great place to go kayaking." "I am looking forward to the holidays." "The lake is a beautiful place to go swimming." "I love to read nonfiction." "The wind is blowing strongly today." "I am grateful for the beauty of nature." "The earth is a fragile and beautiful place." "I love to learn about art."
    "The stars are shining brightly in the sky." "I love to play with my friends." "The rain is falling gently outside." "I am excited for the summer." "The flowers are blooming beautifully." "I love to listen to music." "The thunder is rumbling loudly." "I am looking forward to the weekend." "The sun is setting slowly in the sky." "I love to try new hobbies."
    "The world is a complex and interesting place." "I love to learn about history." "The ocean is a powerful and mysterious place." "I am grateful for my family." "The forest is a great place to go on a hike." "I love to read books." "The desert is a harsh and beautiful environment." "I am excited for the future." "The city is a great place to live." "I love to travel to new cities."
    "The snow is falling gently outside." "I love to play sports in my free time." "The river is a great place to go fishing." "I am looking forward to the holidays." "The lake is a beautiful place to relax." "I love to read poetry." "The wind is blowing strongly today." "I am grateful for the simple things in life." "The earth is a fragile and beautiful place." "I love to learn about science."
    "The stars are shining brightly in the sky." "I love to play with my pets." "The rain is falling gently outside." "I am excited for the summer." "The flowers are blooming beautifully." "I love to listen to podcasts." "The thunder is rumbling loudly." "I am looking forward to the weekend." "The sun is setting slowly in the sky." "I love to try new restaurants."
    "The world is a complex and interesting place." "I love to learn about art." "The ocean is a powerful and mysterious place." "I am grateful for my health." "The forest is a great place to go on an adventure." "I love to read fiction." "The desert is a harsh and beautiful environment." "I am excited for the future." "The city is a great place to meet new people." "I love to travel to new countries."
    "The snow is falling heavily outside." "I love to play board games in my free time." "The river is a great place to go kayaking." "I am looking forward to the holidays." "The lake is a beautiful place to go swimming." "I love to read nonfiction." "The wind is blowing strongly today." "I am grateful for the beauty of nature." "The earth is a fragile and beautiful place." "I love to learn about history."
    "The stars are shining brightly in the sky." "I love to play with my friends." "The rain is falling gently outside." "I am excited for the summer." "The flowers are blooming beautifully." "I love to listen to music." "The thunder is rumbling loudly." "I am looking forward to the weekend." "The sun is setting slowly in the sky." "I love to try new hobbies."
    "The world is a complex and interesting place." "I love to learn about science." "The ocean is a powerful and mysterious place." "I am grateful for my family." "The forest is a great place to go on a hike." "I love to read books." "The desert is a harsh and beautiful environment." "I am excited for the future." "The city is a great place to live." "I love to travel to new cities."
    "The snow is falling gently outside." "I love to play sports in my free time." "The river is a great place to go fishing." "I am looking forward to the holidays." "The lake is a beautiful place to relax." "I love to read poetry." "The wind is blowing strongly today." "I am grateful for the simple things in life." "The earth is a fragile and beautiful place." "I love to learn about art."
    "The stars are shining brightly in the sky." "I love to play with my pets." "The rain is falling gently outside." "I am excited for the summer." "The flowers are blooming beautifully." "I love to listen to podcasts." "The thunder is rumbling loudly." "I am looking forward to the weekend." "The sun is setting slowly in the sky." "I love to try new restaurants."
    "The world is a complex and interesting place." "I love to learn about history." "The ocean is a powerful and mysterious place." "I am grateful for my health." "The forest is a great place to go on an adventure." "I love to read fiction." "The desert is a harsh and beautiful environment." "I am excited for the future." "The city is a great place to meet new people." "I love to travel to new countries."
    "The snow is falling heavily outside." "I love to play board games in my free time." "The river is a great place to go kayaking." "I am looking forward to the holidays." "The lake is a beautiful place to go swimming." "I love to read nonfiction." "The wind is blowing strongly today." "I am grateful for the beauty of nature." "The earth is a fragile and beautiful place." "I love to learn about science."
    "The stars are shining brightly in the sky." "I love to play with my friends." "The rain is falling gently outside." "I am excited for the summer." "The flowers are blooming beautifully." "I love to listen to music." "The thunder is rumbling loudly." "I am looking forward to the weekend." "The sun is setting slowly in the sky." "I love to try new hobbies."
    "The world is a complex and interesting place." "I love to learn about art." "The ocean is a powerful and mysterious place." "I am grateful for my family." "The forest is a great place to go on a hike." "I love to read books." "The desert is a harsh and beautiful environment." "I am excited for the future." "The city is a great place to live." "I love to travel to new cities."
    "The snow is falling gently outside." "I love to play sports in my free time." "The river is a great place to go fishing." "I am looking forward to the holidays." "The lake is a beautiful place to relax." "I love to read poetry." "The wind is blowing strongly today." "I am grateful for the simple things in life." "The earth is a fragile and beautiful place." "I love to learn about history.")
# Create output directory if it doesn't exist
mkdir -p "$output_dir"
# Loop through each image in the input directory
i=0
for img in "$input_dir"/*; do
    
    # Get the base name of the image (e.g., lenna.png)
    base_name=$(basename "$img")
    
    # Select the next message from the messages array
    message=${messages[$i]}
    ((i++))
    echo ${i}
    # Run the stego.py script with the current image and message
    python stego.py --mydct -i "$img" -s "$message" -o "$output_dir/$base_name"
done
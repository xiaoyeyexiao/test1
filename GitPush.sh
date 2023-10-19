#!/bin/bash

while true; do
    echo "Happy GitAutoPush Starting..."
    time=$(date "+%Y-%m-%d %H:%M:%S")

    echo "Choose an option:"
    echo "0 - Quit"
    echo "1 - Commit all changes"
    echo "2 - Commit specify files and folders"
    
    read -p "Enter your choice: " choice

    case $choice in
        0)
            echo "Exiting GitAutoPush..."
            break
            ;;
        1)
            echo "Executing (1)..."
            git add .
            read -t 30 -p "Please enter a commit comment: " msg
            if [ ! "$msg" ]; then
                echo "[commit message]"
                git commit -m ""
            else
                echo "[commit message] $msg"
                git commit -m "$msg"
            fi
            git push
            echo "GitAutoPush Ending..."
            break
            ;;
        2)
            echo "Executing (2)..."
            time=$(date "+%Y-%m-%d %H:%M:%S")

            # Add specify files and folders
            files_to_add=(
                "file1.txt"
                "file2.txt"
            )

            for item in "${files_to_add[@]}"
            do
                git add "$item"
            done

            read -t 30 -p "Please enter a commit comment: " msg
            if [ ! "$msg" ]; then
                echo "[commit message]"
                git commit -m ""
            else
                echo "[commit message] $msg"
                git commit -m "$msg"
            fi
            git push
            echo "GitAutoPush Ending..."
            break
            ;;
        *)
            echo "Invalid choice. Please enter 0, 1, or 2."
            ;;
    esac
done


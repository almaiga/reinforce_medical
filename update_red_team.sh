#!/bin/bash

# Update red_team module with latest medical_team changes

set -e

echo "=========================================="
echo "Updating red_team Module"
echo "=========================================="
echo ""

if [ ! -d "selfplay-redteaming-reference" ]; then
    echo "❌ Error: selfplay-redteaming-reference directory not found"
    echo "Please run ./download_selfplay_redteaming.sh first"
    exit 1
fi

echo "Copying medical_team to red_team..."
rm -rf selfplay-redteaming-reference/red_team
cp -r medical_team selfplay-redteaming-reference/red_team

echo ""
echo "✅ red_team module updated!"
echo ""
echo "Verifying..."
if [ -f "selfplay-redteaming-reference/red_team/__init__.py" ]; then
    echo "✅ __init__.py found"
fi

if [ -f "selfplay-redteaming-reference/red_team/utils.py" ]; then
    echo "✅ utils.py found"
    
    # Check for required functions
    if grep -q "cot_format_check_and_extract" selfplay-redteaming-reference/red_team/utils.py; then
        echo "✅ cot_format_check_and_extract found"
    else
        echo "❌ cot_format_check_and_extract NOT found"
    fi
    
    if grep -q "get_cot_formatting_reward" selfplay-redteaming-reference/red_team/utils.py; then
        echo "✅ get_cot_formatting_reward found"
    else
        echo "❌ get_cot_formatting_reward NOT found"
    fi
    
    if grep -q "get_redteaming_game_reward_general_sum" selfplay-redteaming-reference/red_team/utils.py; then
        echo "✅ get_redteaming_game_reward_general_sum found"
    else
        echo "❌ get_redteaming_game_reward_general_sum NOT found"
    fi
fi

echo ""
echo "=========================================="
echo "✅ Update Complete!"
echo "=========================================="
echo ""

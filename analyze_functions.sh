#!/bin/bash

# Function Analysis Script for PyQSM Project
# This script compares functions between src/ and pyQSM/ directories
# to identify functions that exist only in src/

echo "=== PyQSM Function Analysis ==="
echo "Analyzing functions in src/ vs pyQSM/ directories"
echo ""

# Step 1: Find all Python files in src/ directory that contain function definitions
echo "Step 1: Finding Python files with function definitions in src/"
find /media/penguaman/code/ActualCode/Research/pyQSM/src -name "*.py" -exec grep -l "^def " {} \;
echo ""

# Step 2: Find all Python files in pyQSM/ directory that contain function definitions
echo "Step 2: Finding Python files with function definitions in pyQSM/"
find /media/penguaman/code/ActualCode/Research/pyQSM/pyQSM -name "*.py" -exec grep -l "^def " {} \;
echo ""

# Step 3: Extract all function definitions from src/ directory with file paths
echo "Step 3: Extracting all function definitions from src/ directory"
find /media/penguaman/code/ActualCode/Research/pyQSM/src -name "*.py" -exec grep -H "^def " {} \;
echo ""

# Step 4: Extract all function definitions from pyQSM/ directory with file paths
echo "Step 4: Extracting all function definitions from pyQSM/ directory"
find /media/penguaman/code/ActualCode/Research/pyQSM/pyQSM -name "*.py" -exec grep -H "^def " {} \;
echo ""

# Step 5: Create sorted list of function names from src/ directory
# Extract just the function names (everything before the first parenthesis)
echo "Step 5: Creating sorted list of function names from src/"
find /media/penguaman/code/ActualCode/Research/pyQSM/src -name "*.py" -exec grep -h "^def " {} \; | sed 's/def \([^(]*\).*/\1/' | sort > /tmp/src_functions.txt
echo "Created /tmp/src_functions.txt with $(wc -l < /tmp/src_functions.txt) functions"
echo ""

# Step 6: Create sorted list of function names from pyQSM/ directory
# Extract just the function names (everything before the first parenthesis)
echo "Step 6: Creating sorted list of function names from pyQSM/"
find /media/penguaman/code/ActualCode/Research/pyQSM/pyQSM -name "*.py" -exec grep -h "^def " {} \; | sed 's/def \([^(]*\).*/\1/' | sort > /tmp/pyqsm_functions.txt
echo "Created /tmp/pyqsm_functions.txt with $(wc -l < /tmp/pyqsm_functions.txt) functions"
echo ""

# Step 7: Find functions that exist only in src/ directory
# comm -23 shows lines in first file that are not in second file
echo "Step 7: Finding functions unique to src/ directory"
echo "Functions that exist ONLY in src/ and NOT in pyQSM/:"
comm -23 /tmp/src_functions.txt /tmp/pyqsm_functions.txt
echo ""

# Step 8: Show count of unique functions
echo "Step 8: Summary"
unique_count=$(comm -23 /tmp/src_functions.txt /tmp/pyqsm_functions.txt | wc -l)
echo "Total functions unique to src/: $unique_count"
echo ""

# Step 9: Optional - Show functions that exist in both directories
echo "Step 9: Functions that exist in both directories (for reference):"
comm -12 /tmp/src_functions.txt /tmp/pyqsm_functions.txt | head -10
echo "... (showing first 10, total: $(comm -12 /tmp/src_functions.txt /tmp/pyqsm_functions.txt | wc -l))"
echo ""

# Step 10: Clean up temporary files
echo "Step 10: Cleaning up temporary files"
rm -f /tmp/src_functions.txt /tmp/pyqsm_functions.txt
echo "Analysis complete!"
echo ""

# Additional analysis commands (commented out - uncomment if needed)
# echo "=== Additional Analysis ==="
# echo "Functions unique to pyQSM/ (if any):"
# comm -13 /tmp/src_functions.txt /tmp/pyqsm_functions.txt
# echo ""
# echo "Total functions in src/: $(wc -l < /tmp/src_functions.txt)"
# echo "Total functions in pyQSM/: $(wc -l < /tmp/pyqsm_functions.txt)"

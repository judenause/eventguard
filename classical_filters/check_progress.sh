#!/bin/bash
# Script to check the progress of classical filter evaluation

echo "Checking evaluation progress..."
echo ""

# Check if tmux session exists
if tmux has-session -t classical_eval 2>/dev/null; then
    echo "✅ Tmux session 'classical_eval' is running"
    echo ""
    echo "Last 30 lines of output:"
    echo "========================================"
    tmux capture-pane -t classical_eval -p | tail -30
    echo "========================================"
    echo ""
    echo "To attach to the session: tmux attach -t classical_eval"
    echo "To detach from session: Press Ctrl+B, then D"
else
    echo "❌ Tmux session 'classical_eval' not found"
    echo ""
    if [ -f "evaluation.log" ]; then
        echo "Checking log file..."
        echo "Last 30 lines:"
        echo "========================================"
        tail -30 evaluation.log
        echo "========================================"
    fi
fi

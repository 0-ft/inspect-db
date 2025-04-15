Require inspect-db

# Set up a 1200x600 terminal with 46px font.
Set FontSize 14
Set Width 1000
Set Height 800
Set TypingSpeed 20ms
Set WindowBar Colorful
Set Padding 20
Set Margin 50
Set MarginFill "#18813D"
Set LineHeight 1.2
Set BorderRadius 10
Set Framerate 30

# Set PlaybackSpeed 1.4

Type "inspect-db ingest duckdb:///evals.db tests/sample_logs/*.eval"

Sleep 300ms

Enter

Wait

Sleep 800ms

Type "inspect-db stats duckdb:///evals.db"

Sleep 300ms

Enter

Wait

Sleep 1s

Type "inspect-db grep duckdb:///evals.db -r assistant -p deployed"

Sleep 300ms

Enter

Wait

Sleep 2s


# Terminal demo for the README. Renders docs/assets/demo.gif with VHS
# (https://github.com/charmbracelet/vhs). Every command below really runs.
#
#   docker build --build-arg BASE=debian:bookworm -t framepipe:bookworm .
#   docker build -t framepipe-demo -f docs/Dockerfile.demo .
#   docker run --rm -v "$PWD:/vhs" framepipe-demo docs/demo.tape

Output docs/assets/demo.gif
Require framepipe

Set Shell "bash"
Set FontSize 15
Set Width 1440
Set Height 440
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

Hide
Type "cd /tmp && cp /opt/framepipe/fixtures/random_rgb_257x129.png in.png && clear"
Enter
Show

Type "# Which backends are built, and which are usable right now?"
Enter
Type "framepipe doctor"
Enter
Sleep 2.5s

Hide
Type "clear"
Enter
Show

Type "# Process one image; the JSON report goes to stdout, diagnostics to stderr"
Enter
Type "framepipe process --input in.png --output edges.png --backend cpu-omp --width 128 --height 64 --report - | head -n 9"
Enter
Sleep 3.5s

Type "# Check cpu-omp against the scalar reference, within the documented tolerances"
Enter
Type "framepipe validate --input in.png --candidate cpu-omp --reference cpu-ref --report v.json; echo exit=$?"
Enter
Sleep 3s

Hide
Type "clear"
Enter
Show

Type "# Explicitly requesting CUDA on a CPU-only build fails loudly (exit 4); it never silently falls back"
Enter
Type "framepipe process --input in.png --output gpu.png --backend cuda; echo exit=$?"
Enter
Sleep 4s

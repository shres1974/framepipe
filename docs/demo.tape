# Terminal demo for the README. Renders docs/assets/demo.gif with VHS
# (https://github.com/charmbracelet/vhs). Every command below really runs.
#
#   docker build --build-arg BASE=debian:bookworm -t framepipe:bookworm .
#   python3 scripts/make_showcase.py scene --out build/demo_ctx/scene.png
#   docker build -t framepipe-demo -f docs/Dockerfile.demo build/demo_ctx
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
Type "cd /tmp && cp /opt/framepipe/fixtures/random_rgb_257x129.png in.png && cp /opt/framepipe/samples/scene.png scene.png && clear"
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

Hide
Type "clear"
Enter
Show

Type "# Every backend on one 1080p image: speed, and max error vs the scalar reference"
Enter
Type "framepipe compare --input scene.png --output-dir cmp; ls cmp"
Enter
Sleep 4s

Hide
Type "clear"
Enter
Show

Type "# Explicitly requesting CUDA on a CPU-only build fails loudly (exit 4); it never silently falls back"
Enter
Type "framepipe process --input in.png --output gpu.png --backend cuda; echo exit=$?"
Enter
Sleep 4s

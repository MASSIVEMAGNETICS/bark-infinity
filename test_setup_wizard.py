from bark_infinity.setup_wizard import create_launcher_script


def test_create_launcher_script_linux(tmp_path):
    launcher_path = create_launcher_script(
        mode="webui",
        port=7860,
        share=True,
        output_dir=tmp_path,
        platform_name="Linux",
    )

    assert launcher_path.exists()
    content = launcher_path.read_text(encoding="utf-8")
    assert "[Desktop Entry]" in content
    assert "Name=Bark Infinity Web UI" in content
    assert "-m bark_infinity.cli webui --port 7860 --share" in content

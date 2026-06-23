"""Custom PyInstaller hook for webrtcvad (webrtcvad-wheels).

The upstream hook fails because the wheel does not expose package metadata
under the import name ``webrtcvad``. This empty hook avoids that failure while
still allowing PyInstaller to collect the extension module via hiddenimports.
"""

# No metadata to copy; the _webrtcvad extension is collected automatically.

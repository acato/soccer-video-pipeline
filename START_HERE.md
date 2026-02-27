# Make Your Goalkeeper Reel

Turn your game film into a highlight reel — automatically. You drop in a video of your soccer match, and this app finds every save, goal kick, and big play your goalkeeper made, then cuts them together into one shareable reel.

---

## What You Need

- **A Mac or PC** (Macs with Apple chips — M1, M2, M3, M4 — are the fastest)
- **Docker Desktop** (a free app that runs everything behind the scenes)
- **Your game video** (an MP4 file from your camera, phone, or GoPro)
- **Your team's jersey colors** (you'll set these up once)

---

## Step 1: Install Docker Desktop

Docker Desktop is a free app. It's what runs the video processing for you.

1. Go to [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) and download it
2. Open the installer and follow the steps (just click "Next" / "Agree" / "Install")
3. Open Docker Desktop
4. Wait until you see a **whale icon** in your menu bar (Mac) or system tray (Windows) — that means it's ready

> If it asks you to create an account, you can skip that.

---

## Step 2: Open Your Terminal

You only need the terminal for setup. After that, everything happens in your browser.

- **Mac:** Press `Cmd + Space`, type **Terminal**, and press Enter
- **Windows:** Click the Start menu, type **Terminal** or **PowerShell**, and press Enter

You should see a window with a blinking cursor. That's it — you're in!

---

## Step 3: Download This Project

**Option A — If you have Git installed:**

Copy and paste this into your terminal, then press Enter:

```
git clone https://github.com/YOUR-USERNAME/soccer-video-pipeline.git
cd soccer-video-pipeline
```

**Option B — No Git? No problem:**

1. On this GitHub page, click the green **Code** button, then **Download ZIP**
2. Unzip the file (double-click it)
3. In your terminal, type `cd ` (with a space after it), then drag the unzipped folder into the terminal window and press Enter

---

## Step 4: Set Up the App (One Time)

Run this command:

```
make deploy
```

It will ask you **two questions:**

1. **Where are your game videos?**
   Type the path to the folder where your game recordings are stored.
   - Mac example: `/Volumes/SoccerGames` or `/Users/yourname/Movies/Soccer`
   - Windows example: `D:\SoccerGames`

2. **Where should finished reels go?**
   Type the path to a folder where you want your reels saved.
   - Mac example: `/Users/yourname/Movies/Reels`
   - Windows example: `D:\Reels`

Wait about 2 minutes. When you see **"Stack is up"**, you're done with setup!

---

## Step 5: Set Up Your Team (One Time)

Tell the app what your team looks like so it can find your goalkeeper in the video.

Run this command, but replace the example with your team's info:

```
./setup-team.sh "Seattle Reign FC" --kit Home blue teal --kit Away white neon_yellow
```

Here's what each part means:

| Part | What it is | Example |
|------|-----------|---------|
| `"Seattle Reign FC"` | Your team's name (in quotes) | `"Lightning SC"` |
| `--kit Home` | The name of a jersey set | `Home`, `Away`, `Third` |
| `blue` | The color most of your players wear | `blue` |
| `teal` | The color your goalkeeper wears | `teal` |

You can add as many jersey sets as you want. For example, if your team has three kits:

```
./setup-team.sh "Lightning SC" \
  --kit Home dark_blue neon_yellow \
  --kit Away white neon_green \
  --kit Third black neon_pink
```

You can also add a new jersey set later:

```
./setup-team.sh --add-kit Fourth red neon_orange
```

### What colors can I use?

Run this to see all the available colors:

```
./setup-team.sh --colors
```

Here's the full list:

| Color | |
|-------|---|
| `white` | &#x2B1C; |
| `silver` | &#x25FB;&#xFE0F; |
| `gray` | &#x2B1B; |
| `black` | &#x25FC;&#xFE0F; |
| `red` | &#x1F7E5; |
| `dark_red` | &#x1F7E5; |
| `maroon` | &#x1F7EB; |
| `burgundy` | &#x1F7EB; |
| `orange` | &#x1F7E7; |
| `neon_orange` | &#x1F7E7; |
| `yellow` | &#x1F7E8; |
| `neon_yellow` | &#x1F7E8; |
| `green` | &#x1F7E9; |
| `dark_green` | &#x1F7E9; |
| `neon_green` | &#x1F7E9; |
| `teal` | &#x1F7E6; |
| `sky_blue` | &#x1F7E6; |
| `light_blue` | &#x1F7E6; |
| `blue` | &#x1F7E6; |
| `dark_blue` | &#x1F7E6; |
| `navy` | &#x1F7E6; |
| `purple` | &#x1F7EA; |
| `pink` | &#x1F7EA; |
| `hot_pink` | &#x1F7EA; |
| `neon_pink` | &#x1F7EA; |

---

## Step 6: Open Your Dashboard

Open your browser and go to:

**[http://localhost:8080/ui](http://localhost:8080/ui)**

Bookmark this page! You'll come back here every time you want to process a game.

---

## Step 7: Process a Game

1. In the dashboard, pick your **video file** from the dropdown (it lists every video in your NAS folder)
2. Pick which **jersey** your team was wearing in that game (Home, Away, etc.)
3. Check which reels you want: **Goalkeeper**, **Highlights**, or both (both are checked by default)
4. Set **Game start (min)** if your video has warmup footage before kickoff. For example, if the game starts 5 minutes into the recording, enter `5`. This skips the warmup so it doesn't create false detections. Leave at `0` if the video starts at kickoff.
5. Click **Submit Job**
6. Watch the progress bar move across the screen. You can **Pause** or **Cancel** processing at any time using the buttons in the jobs table.
7. When it hits 100%, click the **download link** to get your reel

That's it! Your goalkeeper reel is ready to share.

> For a detailed walkthrough of every part of the dashboard, see [docs/web-ui.md](docs/web-ui.md).

---

## Troubleshooting

**"I see an error about Docker"**
Is Docker Desktop running? Look for the whale icon in your menu bar or system tray. If it's not there, open Docker Desktop and wait for it to start.

**"It says file not found"**
Make sure your video file is inside the folder you picked in Step 4. The filename you type in the dashboard has to match exactly (including `.mp4`).

**"It's really slow"**
Totally normal! A full game takes 30 to 60 minutes to process. Grab a snack — the progress bar will keep updating so you can check in.

**"How do I stop everything?"**
Run this in your terminal:
```
make down
```

**"How do I start it again next time?"**
Run this in your terminal:
```
make up
```
Then open [http://localhost:8080/ui](http://localhost:8080/ui) in your browser.

**"I want to change my team's colors"**
Just run the setup-team command again with the new colors. It will overwrite the old ones.

**"Can I process a game for a different team?"**
This app is set up for your team only. Each time you submit a game, it uses your saved team colors to find your goalkeeper.

---

## For Advanced Users

If you want to set the opponent team's colors for better accuracy (optional), you can use the full API directly:

```bash
curl -X POST http://localhost:8080/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "nas_path": "saturday_game.mp4",
    "match_config": {
      "team": {"team_name": "Your FC", "outfield_color": "blue", "gk_color": "teal"},
      "opponent": {"team_name": "Other FC", "outfield_color": "red", "gk_color": "neon_yellow"}
    }
  }'
```

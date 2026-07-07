# Deploying NodeSense

Two deployment targets: the backend goes to Hugging Face Spaces, the frontend goes to GitHub Pages. Both are free.

## Backend to Hugging Face Spaces

1. Create an account at huggingface.co
2. Go to huggingface.co/spaces/new and create a new Space
3. Choose Docker as the SDK, and set visibility to Private if you want it professor only
4. Clone the Space repo it gives you
5. Copy everything from the `backend/` folder into the cloned Space directory
6. Commit and push

```bash
git add .
git commit -m "Deploy NodeSense backend"
git push
```

The Space will build and show a Running status when ready. Your API base URL will look like:

```
https://YOUR_USERNAME-nodesense.hf.space
```

Test it by visiting that URL. You should see the health check response.

### Things to remember

- The app must listen on port 7860. This is already set in `app.py` and the `Dockerfile`.
- Only the `/tmp` directory is writable. Model caches are already redirected there in the `Dockerfile`.
- On a public Space, your code and endpoints are visible to anyone. Use a private Space to keep it between you and your advisor — but note the dashboard on GitHub Pages can only call a public Space, so for the professor demo keep the Space public and the GitHub repo is what you can keep private.
- The trained model in `backend/artifacts/` ships with the code, so the Space serves live model predictions immediately — no training on the Space.
- The image installs only `requirements.txt` (no PyTorch), which keeps the Space build small and fast. Training happens locally with `requirements-train.txt`.

## Frontend to GitHub Pages

1. In `frontend/package.json`, set the `homepage` field to your GitHub Pages URL
2. In `frontend/vite.config.js`, set `base` to match your repo name
3. In `frontend/src/App.jsx`, set `API_BASE` and `WS_BASE` to your Space URL

```bash
cd frontend
npm install
npm run deploy
```

This builds the app and pushes it to a `gh-pages` branch. Then in your GitHub repo settings, under Pages, set the source to the `gh-pages` branch. Your dashboard goes live at the homepage URL within a minute.

## Public launch upgrade path

When you want a more polished public version:

- Move the frontend from GitHub Pages to Vercel or Netlify. Connect your repo and it redeploys on every push.
- Keep the backend on Hugging Face Spaces, or move to Render's free tier. Note that free backends sleep after inactivity and take a few seconds to wake on the next request.
- Add a free Postgres database from Supabase or Neon if you want to store alert history.
- Optionally add a custom domain from Cloudflare or Namecheap for about ten dollars a year.

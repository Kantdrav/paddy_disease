# Render Deployment Quick Start Guide

This guide will help you deploy the Paddy Disease Classification API on Render in just a few minutes.

## Prerequisites

- A Render account (free tier available at https://render.com)
- This GitHub repository connected to your Render account

## Step-by-Step Deployment

### 1. Create a New Web Service

1. Log in to your Render dashboard
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository (if not already connected)
4. Select the `Kantdrav/paddy_disease` repository

### 2. Configure the Service

Use these exact settings:

| Setting | Value |
|---------|-------|
| **Name** | `paddy-disease-api` (or your preferred name) |
| **Environment** | `Python 3` |
| **Region** | Choose closest to your users |
| **Branch** | `copilot/fix-python-version-compatibility` (or `main` after merge) |
| **Root Directory** | Leave empty |
| **Build Command** | `pip install -r ai_model/requirements.onnx.txt` |
| **Start Command** | `cd ai_model && bash start.sh` |

### 3. Advanced Settings (Optional)

Click **"Advanced"** to configure:

- **Python Version**: Will be automatically set to `3.12.3` via the `.python-version` file in `ai_model/`
- **Instance Type**: Free tier is sufficient for testing
- **Auto-Deploy**: Enable to automatically deploy on git push

### 4. Deploy

1. Click **"Create Web Service"**
2. Wait for the build to complete (2-3 minutes)
3. Your API will be available at `https://your-service-name.onrender.com`

## Testing Your Deployment

Once deployed, test your API:

### Health Check
```bash
curl https://your-service-name.onrender.com/
```

Expected response:
```json
{
  "message": "Paddy Disease API (ONNX Runtime)",
  "classes": ["Bacterial Leaf Blight", "Brown Spot"],
  "num_classes": 2
}
```

### Make a Prediction
```bash
curl -X POST "https://your-service-name.onrender.com/predict/" \
  -F "file=@/path/to/your/image.jpg"
```

Expected response:
```json
{
  "prediction": "Bacterial Leaf Blight"
}
```

## What Just Happened?

The deployment uses:

- **ONNX Runtime** instead of PyTorch for lightweight inference
- **Python 3.12.3** (compatible with all dependencies)
- **~150 MB total size** (vs ~1.5 GB with PyTorch)
- **Existing ONNX model** files (`model.onnx` and `model.onnx.data`)

## Troubleshooting

### Build Failed

**Common causes:**

1. **Wrong Python version**: Make sure `.python-version` file exists in `ai_model/` directory
2. **Missing model files**: Ensure `model.onnx`, `model.onnx.data`, and `model.classes.json` are committed to the repository
3. **Wrong build command**: Use exactly `pip install -r ai_model/requirements.onnx.txt`

### Service Won't Start

**Common causes:**

1. **Wrong start command**: Use exactly `cd ai_model && bash start.sh`
2. **Port binding issues**: The start script automatically uses Render's `PORT` environment variable
3. **Missing dependencies**: Check the build logs for any failed installations

### Prediction Errors

**Common causes:**

1. **Model not loaded**: Check service logs for "Model loaded successfully" message
2. **Invalid image format**: Make sure you're uploading a valid image file (JPG, PNG, etc.)
3. **Wrong endpoint**: Use `/predict/` (with trailing slash)

## Monitoring and Logs

- **Logs**: Available in the Render dashboard under your service
- **Metrics**: View request counts, response times, and errors
- **Health Check**: Use the `/health` endpoint for monitoring

## Updating Your Deployment

To deploy changes:

1. Push changes to your GitHub repository
2. Render will automatically rebuild and deploy (if auto-deploy is enabled)
3. Manual deploy: Click **"Manual Deploy"** → **"Deploy latest commit"** in Render dashboard

## Cost Optimization

**Free Tier:**
- Render's free tier includes 750 hours/month
- Service spins down after 15 minutes of inactivity
- First request after spin-down takes ~30 seconds

**Paid Tier Benefits:**
- No spin-down
- Faster cold starts
- More CPU/memory
- Custom domains

## Next Steps

1. **Add Custom Domain**: Configure a custom domain in Render settings
2. **Set Up Monitoring**: Use Render's built-in monitoring or integrate with external tools
3. **Scale**: Upgrade to a paid plan for better performance and no spin-down
4. **CI/CD**: Enable auto-deploy for automatic deployments on git push

## Support

For more detailed information:
- See [DEPLOYMENT.md](DEPLOYMENT.md) for all deployment options
- See [README.md](README.md) for API documentation
- Check Render documentation: https://render.com/docs

## Success! 🎉

Your Paddy Disease Classification API is now live and ready to use!

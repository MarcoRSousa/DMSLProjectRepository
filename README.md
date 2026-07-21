# DMSL Hybrid Manga Recommendation Framework

Manga are comic books and graphic novels originating from Japan, known for their distinctive art style, expressive characters, and wide variety of genres. 

This project develops and evaluates a hybrid manga recommendation framework that combines content-based and collaborative filtering techniques to recommend similar manga.


# Author
**Marco Sousa**
**ISYE 7406 (DMSL): Team 067**

# Running the application

The only requirement is [Docker Desktop](https://www.docker.com/products/docker-desktop/).

Here are the steps in order to use the application on first run:

Build the Docker image:

```bash
git clone <repo-url>
cd AMGBase
docker compose up --build
```

The first run takes quite some time. Docker first has to build the image.  The sentence-transformers package uses large models to construct semantic embeddings, and thus causes the image to likewise be quite large.

Fully fetching and processing the data is sinificantly time consuming. As such, the data folder has already been pre-populated with content-based, collaborative, and hybrid graph data.

Running example recomennations run through the recommender folder:

```bash
docker compose exec backend python -m recommendations.recommender 
```

Running model validation runs though the recommender_evaluation file.

```bash
docker compose exec backend python -m recommendations.recommender_evaluation
```

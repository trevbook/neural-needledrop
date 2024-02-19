FROM node:18.13.0-alpine

WORKDIR /app

# Add the build argument for the API base URL
ARG VITE_API_BASE_URL

# Set the environment variable
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

COPY package.json .

RUN npm install

COPY . .

RUN npm run build

EXPOSE 8080

CMD ["npm", "run", "preview"]

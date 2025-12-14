# Dockerfile for Elasticsearch BM25 service
FROM docker.elastic.co/elasticsearch/elasticsearch:8.11.0

# Configure Elasticsearch for development
ENV discovery.type=single-node
ENV xpack.security.enabled=false
ENV ES_JAVA_OPTS="-Xms512m -Xmx512m"

EXPOSE 9200 9300

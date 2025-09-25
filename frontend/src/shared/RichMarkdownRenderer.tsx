import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';

// Utility functions for content detection
export const isYouTubeUrl = (url: string): boolean => {
  const youtubeRegex = /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/;
  return youtubeRegex.test(url);
};

export const getYouTubeVideoId = (url: string): string | null => {
  const youtubeRegex = /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/;
  const match = url.match(youtubeRegex);
  return match ? match[1] : null;
};

export const isImageUrl = (url: string): boolean => {
  const imageExtensions = /\.(jpg|jpeg|png|gif|bmp|webp|svg)$/i;
  return imageExtensions.test(url);
};

export const isGifUrl = (url: string): boolean => {
  return /\.(gif)$/i.test(url);
};

// Custom components for rich content
const LinkRenderer: React.FC<{ href?: string; children?: React.ReactNode }> = ({ href, children }) => {
  if (!href) return <>{children}</>;

  // Check if it's a YouTube URL
  if (isYouTubeUrl(href)) {
    const videoId = getYouTubeVideoId(href);
    if (videoId) {
      return (
        <div className="youtube-embed">
          <iframe
            width="560"
            height="315"
            src={`https://www.youtube.com/embed/${videoId}`}
            title="YouTube video player"
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          ></iframe>
        </div>
      );
    }
  }

  // Check if it's an image/GIF
  if (isImageUrl(href)) {
    return (
      <div className="image-embed">
        <img
          src={href}
          alt={typeof children === 'string' ? children : 'Embedded image'}
          onError={(e) => {
            console.error('Failed to load image:', href);
            e.currentTarget.style.display = 'none';
          }}
        />
      </div>
    );
  }

  // Regular link
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="markdown-link"
    >
      {children}
    </a>
  );
};

const ImageRenderer: React.FC<{ src?: string; alt?: string }> = ({ src, alt }) => {
  if (!src) return null;

  return (
    <div className="image-embed">
      <img
        src={src}
        alt={alt || 'Embedded image'}
        onError={(e) => {
          console.error('Failed to load image:', src);
          e.currentTarget.style.display = 'none';
        }}
      />
    </div>
  );
};

const CodeBlockRenderer: React.FC<{ children?: React.ReactNode; className?: string }> = ({ children, className }) => {
  const language = className?.replace('language-', '') || '';
  return (
    <pre className={`code-block ${language ? `language-${language}` : ''}`}>
      <code className={className}>
        {children}
      </code>
    </pre>
  );
};

interface RichMarkdownRendererProps {
  content: string;
  className?: string;
}

const RichMarkdownRenderer: React.FC<RichMarkdownRendererProps> = ({ content, className = '' }) => {
  return (
    <div className={`rich-markdown ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight, rehypeRaw]}
        components={{
          a: LinkRenderer,
          img: ImageRenderer,
          pre: CodeBlockRenderer,
          code: ({ children, className }) => {
            const isInline = !className?.includes('language-');
            return (
              <code className={isInline ? 'inline-code' : className}>
                {children}
              </code>
            );
          }
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default RichMarkdownRenderer;

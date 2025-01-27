import Markdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { materialDark } from "react-syntax-highlighter/dist/cjs/styles/prism";
import { useSearchParams } from "react-router";
import "./MDFile.css";
import TOC from "../components/TOC";
import useFetchMdContent from "../hooks/useFetchMdContent";
import renderHeading from "../utils/renderHeading";

type ExtraProps = {
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
};

const MDFile = () => {
  const [searchParams] = useSearchParams();
  const title = searchParams.get("title");
  const date = searchParams.get("date");
  const { content, fileName } = useFetchMdContent();

  return (
    <div className="markdown-container">
      <div className="markdown-content">
        <div className="md-header">
          <div className="md-title">{title}</div>
          <div className="md-date">{date}</div>
        </div>

        <Markdown
          components={{
            h1: ({ children }) => renderHeading({ children, headingNo: 1 }),
            h2: ({ children }) => renderHeading({ children, headingNo: 2 }),
            h3: ({ children }) => renderHeading({ children, headingNo: 3 }),
            code({ inline, className, children, ...props }: ExtraProps) {
              const match = /language-(\w+)/.exec(className || "");
              return !inline && match ? (
                <SyntaxHighlighter
                  style={materialDark}
                  language={match[1]}
                  PreTag="div"
                  {...props}
                >
                  {String(children).replace(/\n$/, "")}
                </SyntaxHighlighter>
              ) : (
                <code className={className} {...props}>
                  {children}
                </code>
              );
            },
          }}
        >
          {content}
        </Markdown>
      </div>
      <div className="toc-container">
        <TOC fileName={fileName} />
      </div>
    </div>
  );
};

export default MDFile;

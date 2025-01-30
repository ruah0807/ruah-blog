import Markdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { materialDark } from "react-syntax-highlighter/dist/cjs/styles/prism";
import { useSearchParams } from "react-router";
import styles from "./MDFile.module.scss";
import TOC from "../components/TOC";
import useFetchMdContent from "../hooks/useFetchMdContent";
import renderHeading from "../utils/renderHeading";
import rehypeRaw from "rehype-raw";
import remarkGfm from "remark-gfm";

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
    <div className={styles.markdown_container}>
      <div className={styles.markdown_content}>
        <div className={styles.md_header}>
          <div className={styles.md_title}>{title}</div>
          <div className="md-date">{date}</div>
        </div>

        <Markdown
          rehypePlugins={[rehypeRaw]}
          remarkPlugins={[remarkGfm]}
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

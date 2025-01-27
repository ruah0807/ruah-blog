import { useEffect, useState } from "react";
import matter from "gray-matter";
import { useNavigate } from "react-router";
import getDateString from "../utils/getDateString";
import styles from "./PostList.module.scss";
type Post = {
  title: string;
  date: string;
  fileName: string;
  subtitle: string;
  category: string | null;
};

const PostList = () => {
  const [state, setState] = useState({
    postsByCategory: {} as Record<string, Post[]>,
    selectedCategory: null as string | null,
    selectedPost: "",
  });
  const navigate = useNavigate();

  useEffect(() => {
    const importMarkdownFiles = async () => {
      const files = import.meta.glob("/posts/**/*.md", {
        query: "?raw",
        import: "default",
      });
      const postList: Post[] = [];

      for (const path in files) {
        const content = await files[path]();
        const { data } = matter(content as string);
        const parts = path.split("/");
        const fileName = parts.slice(2).join("/"); // Include category if present
        const date = getDateString(fileName);
        const category = parts.length > 3 ? parts[2] : null;
        const subtitle = category
          ? `${category}/${fileName
              .replace(".md", "")
              .split("/")
              .slice(1)
              .join("/")}`
          : fileName.replace(".md", "");
        postList.push({
          title: data.title,
          date,
          fileName,
          subtitle,
          category,
        });
      }
      const groupedPosts = postList.reduce((acc, post) => {
        const key = post.category || "Others";
        if (!acc[key]) {
          acc[key] = [];
        }
        acc[key].push(post);
        return acc;
      }, {} as Record<string, Post[]>);

      setState((preState) => ({ ...preState, postsByCategory: groupedPosts }));
    };

    importMarkdownFiles();
  }, []);

  return (
    <div className={styles.sidebar}>
      <p className={styles.sidebar_title}> 블로그 리스트 </p>
      <p className={styles.sidebar_count}>
        전체 블로그 {Object.values(state.postsByCategory).flat().length}개
      </p>
      {Object.entries(state.postsByCategory)
        .sort(([a], [b]) => (a === "Others" ? 1 : b === "Others" ? -1 : 0))
        .map(([category, posts]) => (
          <div className={styles.sidebar_category} key={category}>
            {category !== "Others" && (
              <div
                className={styles.sidebar_folder}
                draggable="false"
                onClick={() =>
                  setState((prevState) => ({
                    ...prevState,
                    selectedCategory:
                      prevState.selectedCategory === category ? null : category,
                  }))
                }
              >
                {state.selectedCategory === category ? "▼ " : "▷ "} {category}
              </div>
            )}
            {(state.selectedCategory === category || category === "Others") && (
              <ul className={styles.sidebar_list}>
                {posts.map((post, index) => {
                  console.log("포스트", post);
                  return (
                    <li
                      key={index}
                      onClick={() => {
                        navigate(
                          `/${post.subtitle}?title=${post.title}&date=${post.date}`
                        );
                        setState((prevState) => ({
                          ...prevState,
                          selectedPost: post.fileName,
                        }));
                      }}
                      className={`${styles.file_item} ${
                        state.selectedPost === post.fileName
                          ? styles.active
                          : ""
                      }`}
                    >
                      {post.title}
                      <span className={styles.sidebar_date}> {post.date}</span>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        ))}
    </div>
  );
};

export default PostList;

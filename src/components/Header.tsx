import React from 'react';

const Header: React.FC = () => {
  return (
    <header>
      <h1 className='header-title'>롸그</h1>
      <nav>
        <ul>
          <li>Home</li>
          <li>Posts</li>
          <li>About Me</li>
        </ul>
      </nav>
    </header>
  );
};

export default Header; 